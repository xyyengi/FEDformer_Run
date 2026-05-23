import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from exp.exp_logger import ExperimentLogger
from models import FEDformer, Autoformer, Informer, Transformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric


warnings.filterwarnings('ignore')


class SolarConstrainedLoss(nn.Module):
    """
    带 Solar 物理约束的损失函数
    1. 基础损失: Huber Loss (对波峰波谷更敏感)
    2. 负值惩罚: Solar 预测值 < 0 时额外惩罚
    3. 可选: 波峰波谷位置加权
    """
    def __init__(self, solar_idx=1, negative_penalty=0.5, delta=1.0):
        super(SolarConstrainedLoss, self).__init__()
        self.solar_idx = solar_idx  # Solar 在特征中的索引
        self.negative_penalty = negative_penalty
        self.huber = nn.HuberLoss(delta=delta, reduction='none')
    
    def forward(self, pred, true):
        # 基础 Huber Loss
        base_loss = self.huber(pred, true)
        
        # Solar 负值惩罚
        solar_pred = pred[:, :, self.solar_idx]
        negative_mask = (solar_pred < 0).float()
        negative_penalty = self.negative_penalty * torch.abs(solar_pred) * negative_mask
        
        # 总损失
        total_loss = base_loss.mean() + negative_penalty.mean()
        return total_loss


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        # 初始化实验日志器
        self.logger = None
        self.feature_names = ['Wind', 'Solar', 'Load']  # 默认特征名称

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        if self.args.output_attention:
            outputs = outputs[0]

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        return outputs, batch_y

    def _inverse_if_needed(self, data_set, values):
        if not hasattr(data_set, 'inverse_transform'):
            return values

        if self.args.features in ['M', 'S']:
            original_shape = values.shape
            values = data_set.inverse_transform(values.reshape(-1, original_shape[-1]))
            return values.reshape(original_shape)

        if self.args.features == 'MS' and values.shape[-1] == 1 and hasattr(data_set, 'scaler'):
            return values * data_set.scaler.scale_[-1] + data_set.scaler.mean_[-1]

        return values

    def _collect_split_predictions(self, setting, flag, load=False, drop_last=False):
        split_data, split_loader = self._get_data(flag, drop_last=drop_last)

        if load:
            checkpoint_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in split_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = self._inverse_if_needed(split_data, preds)
        trues = self._inverse_if_needed(split_data, trues)
        residual_actual_minus_forecast = trues - preds
        return preds, trues, residual_actual_minus_forecast

    def export_split_predictions(self, setting, load=True):
        folder_path = os.path.join('./results', setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for split_name in ['train', 'val', 'test']:
            preds, trues, residual_actual_minus_forecast = self._collect_split_predictions(
                setting=setting,
                flag=split_name,
                load=load,
                drop_last=False,
            )
            np.save(os.path.join(folder_path, f'{split_name}_pred.npy'), preds)
            np.save(os.path.join(folder_path, f'{split_name}_true.npy'), trues)
            np.save(os.path.join(folder_path, f'{split_name}_res.npy'), residual_actual_minus_forecast)
            np.save(os.path.join(folder_path, f'{split_name}_forecast.npy'), preds)
            np.save(os.path.join(folder_path, f'{split_name}_actual.npy'), trues)
            np.save(os.path.join(folder_path, f'{split_name}_residual.npy'), residual_actual_minus_forecast)

        return

    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, drop_last=None):
        data_set, data_loader = data_provider(self.args, flag, drop_last=drop_last)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """选择损失函数，支持 MSE、Huber Loss 和带负值惩罚的损失"""
        loss_type = self.args.loss if hasattr(self.args, 'loss') else 'mse'
        
        if loss_type == 'mse':
            criterion = nn.MSELoss()
        elif loss_type == 'huber':
            criterion = nn.HuberLoss(delta=1.0)  # Huber Loss 对异常值更鲁棒
        elif loss_type == 'solar_constrained':
            criterion = SolarConstrainedLoss(
                solar_idx=self.args.solar_idx if hasattr(self.args, 'solar_idx') else 1,
                negative_penalty=0.5
            )
        else:
            criterion = nn.MSELoss()
        
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        # 初始化实验日志器（带时间戳）
        from datetime import datetime
        exp_name = f"{setting}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = ExperimentLogger(base_dir='./experiments', exp_name=exp_name)
        self.logger.log_args(self.args)
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
            delta=self.args.early_stop_delta,
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        current_lr = self.args.learning_rate

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 5 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            
            # 记录到日志
            if self.logger:
                self.logger.log_train_step(epoch + 1, train_loss, vali_loss, current_lr)
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            # 更新当前学习率
            current_lr = model_optim.param_groups[0]['lr']

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        # 绘制训练曲线
        if self.logger:
            self.logger.plot_training_curves()

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape before inverse:', preds.shape, trues.shape)
        
        # INVERSE TRANSFORM
        # Check if inverse transform is needed for true scale metric evaluation
        if hasattr(test_data, 'inverse_transform') and (self.args.features == 'M' or self.args.features == 'S'):
            preds = test_data.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape)
            trues = test_data.inverse_transform(trues.reshape(-1, trues.shape[-1])).reshape(trues.shape)
            print('Applied inverse transform.')
        
        print('test shape after inverse:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        
        # 使用日志系统保存预测结果和可视化
        if self.logger:
            # 记录指标
            self.logger.log_metrics({
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'mspe': float(mspe)
            }, split='test')
            
            # 保存预测结果和可视化（只取前3个特征：Wind, Solar, Load）
            if preds.shape[-1] >= 3:
                preds_3feat = preds[:, :, :3]
                trues_3feat = trues[:, :, :3]
                self.logger.save_predictions(preds_3feat, trues_3feat, split='test', 
                                           feature_names=self.feature_names)
            
            # 生成总结报告
            self.logger.generate_summary_report()

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.concatenate(preds, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
