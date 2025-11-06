import copy
import os
import gradio as gr
import blobfile as bf
import torch as th
from torch.optim import AdamW

from . import logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from PIL import Image
from guided_diffusion.train_test_helper import save_images2pil, summary_metrics
import numpy as np
import pandas as pd

from mappingknow.preswinunet import preSwinUnet

th.serialization.add_safe_globals([preSwinUnet])
import torch
import shutil

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
            self,
            *,
            model,
            mode,
            diffusion,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            output=None,
    ):
        self.model = model
        self.mode = mode
        self.diffusion = diffusion
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        self.ema_params = [
            copy.deepcopy(self.mp_trainer.master_params)
            for _ in range(len(self.ema_rate))
        ]
        self.output = output
        self.mapmodel = None
        unique_labels = [0, 10, 20, 30, 40, 50, 60, 70]
        self.num_classes = len(unique_labels)
        # Create mapping
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.mapping = torch.zeros(max(unique_labels) + 1, dtype=torch.long)
        for idx, label in enumerate(unique_labels):
            self.mapping[label] = idx
        self.modelpath = output + '/model'
        if os.path.exists(self.modelpath):
            shutil.rmtree(self.modelpath)
        os.makedirs(self.modelpath)

    def labels_to_onehot(self, labels):
        """
        Convert label tensor to one-hot encoding
        Input: (B, 1, H, W) with values from unique_labels
        Output: (B, num_classes, H, W) one-hot encoded
        """
        B, _, H, W = labels.shape
        labels_2d = labels.squeeze(1)  # (B, H, W)
        # Map labels to indices
        indices = self.mapping[labels_2d.long()]  # (B, H, W)
        # Convert to one-hot
        onehot = torch.nn.functional.one_hot(indices, num_classes=self.num_classes)  # (B, H, W, num_classes)
        onehot = onehot.permute(0, 3, 1, 2).float()  # (B, num_classes, H, W)
        return onehot

    def evaluatebatch(self, batch, bid, addflag='data', mapmodel=None, ntarget=None, device=None):
        self.model.eval()
        total_samples = 0
        batch_size = len(batch['sar'])  # assuming batch[0] is your data tensor
        if ntarget is not None:
            lastpos = ntarget if batch_size > ntarget else batch_size
        else:
            lastpos = batch_size
        sar = batch['sar'][:lastpos].to(device)
        optical = batch['optical'][:lastpos].to(device)
        textcode = batch['labeltext'][:lastpos].to(device)
        files = batch['filename'][:lastpos]
        label = None
        preconditions = sar
        if self.mode == 'SARLABEL' or self.mode == 'SARMAPLABEL':
            label = batch['label'][:lastpos].to(device)
        if mapmodel is not None:
            with th.no_grad():
                mapinf = mapmodel(sar[:lastpos])
            preconditions = th.cat((sar, mapinf), dim=1)
        if self.mode == 'SARLABELonehot':
            onehot = self.labels_to_onehot(batch['label'][:lastpos]).to(device)
            preconditions = th.cat((sar, onehot), dim=1)
        prediction = self.predict(device, preconditions, textcode, label=label)
        flag = [addflag + '_batch' + str(bid) + '_' + afile for afile in
                files]  # batch['filename'][0] + '_' + addflag + '_batch' + str(bid)
        save_images2pil(prediction, optical, sar, self.output, flag)
        flag = addflag + '_batch' + str(bid)
        res = summary_metrics(prediction, optical, self.output, flag, files=files)
        return res

    def predict_dataload(self, dataload, addflag='data', mapmodel=None, ntarget=None, device=None, deftext=False):
        self.model.eval()
        total_samples = 0
        tprediction, toptical = [], []
        tfiles = []
        if deftext:
            fl = '/devb/sar2opt_diff_txt/test/deftextcode.npy'
            textfeature = np.load(fl)  # numpy array, shape: (feature_dim,)
            # Convert to PyTorch tensor and move to device
            textfeature = torch.from_numpy(textfeature).to(device)
        print('starting ... ... ')
        print(f'size: ', len(dataload))
        for bid, batch in enumerate(tqdm(dataload)):
            batch_size = len(batch['sar'])  # assuming batch[0] is your data tensor
            print('batch 1:', batch['sar'].shape)
            if ntarget is not None and total_samples + batch_size > ntarget:
                lastpos = ntarget - total_samples
            else:
                lastpos = batch_size
            print('lastpos:', lastpos)
            sar = batch['sar'][:lastpos].to(device)
            optical = batch['optical'][:lastpos].to(device)
            if deftext:
                # Broadcast to batch size
                textcode = textfeature.unsqueeze(0).repeat(lastpos, 1, 1)
            else:
                textcode = batch['labeltext'][:lastpos].to(device)
            files = batch['filename'][:lastpos]
            label = None
            preconditions = sar
            if self.mode == 'SARLABEL' or self.mode == 'SARMAPLABEL':
                label = batch['label'][:lastpos].to(device)
            if mapmodel is not None:
                with th.no_grad():
                    mapinf = mapmodel(sar[:lastpos])
                preconditions = th.cat((sar, mapinf), dim=1)
            if self.mode == 'SARLABELonehot':
                onehot = self.labels_to_onehot(batch['label'][:lastpos]).to(device)
                preconditions = th.cat((sar, onehot), dim=1)
            print('starting predict... ... ')
            print('preconditions:', preconditions.shape)
            print('textcode:', textcode.shape)
            prediction = self.predict(device, preconditions, textcode, label=label)
            print('end predict... ... ')
            tprediction.append(prediction)
            toptical.append(optical)
            tfiles.extend(files)
            # flag = batch['filename'][0] + '_' + addflag + '_batch' + str(bid)
            flag = [addflag + '_batch' + str(bid) + '_' + afile for afile in files]
            save_images2pil(prediction, optical, sar, self.output, flag, label=None)
        tprediction = th.cat(tprediction, dim=0)
        toptical = th.cat(toptical, dim=0)
        #        flag =  addflag + '_summary_metric'
        flag = addflag + '_batch' + str(bid)
        print(f"{len(tprediction)} vs. {len(toptical)} vs. {len(tfiles)}")
        res = summary_metrics(tprediction, toptical, self.output, flag, files=tfiles)
        return res

    def evaluate(self, dataload, addflag='data', mapmodel=None, ntarget=None, device=None):
        self.model.eval()
        total_samples = 0
        tprediction, toptical = [], []
        for bid, batch in enumerate(tqdm(dataload)):
            batch_size = len(batch['sar'])  # assuming batch[0] is your data tensor
            if ntarget is not None and total_samples + batch_size > ntarget:
                lastpos = ntarget - total_samples
            else:
                lastpos = batch_size
            sar = batch['sar'][:lastpos].to(device)
            optical = batch['optical'][:lastpos].to(device)
            textcode = batch['labeltext'][:lastpos].to(device)
            label = None
            preconditions = sar
            if self.mode == 'SARLABEL' or self.mode == 'SARMAPLABEL':
                label = batch['label'][:lastpos].to(device)
            if mapmodel is not None:
                with th.no_grad():
                    mapinf = mapmodel(sar[:lastpos])
                preconditions = th.cat((sar, mapinf), dim=1)
            if self.mode == 'SARLABELonehot':
                onehot = self.labels_to_onehot(batch['label'][:lastpos]).to(device)
                preconditions = th.cat((sar, onehot), dim=1)
            prediction = self.predict(device, preconditions, textcode, label=label)
            tprediction.append(prediction)
            toptical.append(optical)
            flag = batch['filename'][0] + '_' + addflag + '_batch' + str(bid)
            save_images2pil(prediction, optical, sar, self.output, flag, label=batch['label'])
        tprediction = th.cat(tprediction, dim=0)
        toptical = th.cat(toptical, dim=0)
        flag = addflag + '_summary_metric'
        res = summary_metrics(tprediction, toptical, self.output, flag)
        return res

    def run_train(self, train_dataload, val_dataload, mapath=None, num_epochs=2, device=None):
        self.model.to(device)
        self.step = 0
        tloss = []
        mapmodel = None
        if mapath is not None:
            mapmodel = th.load(mapath, map_location=device, weights_only=False)
        for epoch in range(num_epochs):
            aepoch_loss = []
            for bid, batch in enumerate(tqdm(train_dataload, desc=f'Epoch {epoch + 1}/{num_epochs}')):
                # Get data
                self.model.train()
                sar = batch['sar'].to(device)
                optical = batch['optical'].to(device)
                textcode = batch['labeltext'].to(device)
                label = None
                if self.mode == 'SARLABEL' or self.mode == 'SARMAPLABEL':
                    label = batch['label'].to(device)
                conditions = sar
                if mapath is not None:
                    with th.no_grad():
                        mapinf = mapmodel(sar)
                    conditions = th.cat((sar, mapinf), dim=1)
                if self.mode == 'SARLABELonehot':
                    onehot = self.labels_to_onehot(batch['label']).to(device)
                    conditions = th.cat((sar, onehot), dim=1)
                self.mp_trainer.zero_grad()
                t, weights = self.schedule_sampler.sample(optical.shape[0], device)
                losses = self.diffusion.training_losses(self.model, optical, conditions, t, textcode, label)
                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )
                loss = (losses["loss"] * weights).mean()
                log_loss_dict(
                    self.diffusion, t, {k: v * weights for k, v in losses.items()}
                )
                self.mp_trainer.backward(loss)
                took_step = self.mp_trainer.optimize(self.opt)
                if took_step:
                    self._update_ema()
                self._anneal_lr()
                self.log_step()
                if self.step % self.log_interval == 0:
                    logger.dumpkvs()
                if self.step % self.save_interval == 0:
                    print('w_m:', self.model.con_weight)
                    ntrain, nval = 32, 32
                    addflag = 'step_' + str(self.step) + 'train'
                    self.evaluatebatch(batch, bid, addflag, mapmodel, ntarget=ntrain, device=device)
                    # addflag='step_'+str(self.step)+'val'
                    # self.evaluate(val_dataload, addflag, mapmodel=mapmodel, ntarget=nval, device=device)
                    self.save()
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
                aepoch_loss.append(loss.detach().cpu().numpy())
            print('epoch loss:', np.mean(aepoch_loss))
            tloss.append(np.mean(aepoch_loss))
            batchindex = [i for i in range(len(aepoch_loss))]
            alossDF = pd.DataFrame({'batch': batchindex,
                                    'loss': aepoch_loss}, index=batchindex)
            alossDF.to_csv(self.output + '/tloss_epoch' + str(epoch) + '.csv', index=False)
            epoches = [i for i in range(1, len(tloss) + 1)]
            lossDF = pd.DataFrame({'epoch': epoches, 'loss': tloss},
                                  index=epoches)
            lossDF.to_csv(self.output + '/training_loss.csv', index=False)

    def predict(self, device, condition, textcode, label=None, use_ddim=False):
        self.model.to(device)
        self.model.eval()
        sample_fn = (
            self.diffusion.p_sample_loop if not use_ddim else self.diffusion.ddim_sample_loop
        )
        condition = condition.to(device)
        sample = sample_fn(
            self.model,
            (condition.shape[0], 3, 224, 224),
            textcode,
            label=label,
            clip_denoised=True,
            noise=None,
            condition=condition,
            progress=False
        )
        return sample

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if not rate:
                filename = f"model{(self.step + self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(self.modelpath, filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)
        with bf.BlobFile(
                bf.join(self.modelpath, f"opt{(self.step + self.resume_step):06d}.pt"),
                "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    # '/devb/WHU-OPT-SAR/guideddiff_test1'
    return logger.get_dir()

def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None

def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
