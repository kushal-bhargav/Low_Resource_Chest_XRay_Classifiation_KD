# =====================================================
# Global Configuration Flags
# =====================================================
TASK_TYPE = "multiclass"  # options: "multiclass" or "multilabel"
TEACHER_IMAGE_SIZE = 224
STUDENT_IMAGE_SIZE = 28 # Can change between 28,56,112 
FRACTION = 0.1  # if FRACTION==1.0, forgetting tracker is skipped

# =====================================================
# Import Libraries
# =====================================================
import os
import json
import math
import random
import copy
import heapq
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset

import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
from torchvision.models import vgg19, resnet18, resnet34, resnet50

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, average_precision_score, precision_recall_curve,
    hamming_loss, coverage_error, label_ranking_average_precision_score,
    classification_report
)
from sklearn.preprocessing import label_binarize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# Model Selection Flag
# =====================================================
model_choices = ['vgg19', 'resnet18', 'resnet34', 'resnet50', 'ViT']
model_id = 4  # 0: vgg19, 1: resnet18, 2: resnet34, 3: resnet50, 4: ViT
model_name = model_choices[model_id]
print(f"Selected model: {model_name}")

# =====================================================
# Helper Functions for CSV Processing
# =====================================================
def delete_columns(df, columns_to_keep, root_folder):
    return df[columns_to_keep]

def remove_all_zeros(df, diseases, root_folder):
    mask = df[diseases].sum(axis=1) > 0
    return df[mask]

def add_file_path_column(df, root_folder):
    df["image_id"] = df["image_id"].astype(str)
    df["file_path"] = df["image_id"].apply(lambda x: os.path.join(root_folder, f"{x}.png"))
    return df

# =====================================================
# Forgetting Tracker Utilities
# =====================================================
def initialize_forgetting_tracker(num_samples):
    return {'correct': np.zeros(num_samples, dtype=int),
            'last_seen': np.full(num_samples, -1)}

def update_forgetting_tracker(forgetting_tracker, true_labels, pred_labels, epoch, task_type="multiclass"):
    if task_type == "multilabel":
        correct_predictions = np.all(true_labels == pred_labels, axis=1)
    else:
        correct_predictions = (true_labels == pred_labels)
    for idx, correct in enumerate(correct_predictions):
        if not correct and forgetting_tracker['correct'][idx] > 0:
            forgetting_tracker['last_seen'][idx] = epoch
        forgetting_tracker['correct'][idx] += correct

# =====================================================
# Compute Class Weights for Balanced Loss
# =====================================================
def compute_class_weights(df, task_type, label_column=None):
    if task_type == "multiclass":
        label_counts = df[label_column].value_counts().sort_index()
    elif task_type == "multilabel":
        label_counts = df.sum(axis=0)
    total_samples = label_counts.sum()
    class_weights = total_samples / (len(label_counts) * label_counts)
    class_weights = torch.tensor(class_weights.values, dtype=torch.float32, device=device)
    return class_weights

# =====================================================
# Vision Transformer (ViT) Model Definitions
# =====================================================
class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.embed = config["embed_dim"]
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(self.num_channels, self.embed,
                                    kernel_size=self.patch_size, stride=self.patch_size)
    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["embed_dim"]))
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.patch_embeddings.num_patches + 1, config["embed_dim"])
        )
        self.dropout = nn.Dropout(config["dropout_val"])
    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings.to(x.device)
        x = self.dropout(x)
        return x

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, dropout, bias=True):
        super().__init__()
        self.query = nn.Linear(embed_dim, head_dim, bias=bias)
        self.key = nn.Linear(embed_dim, head_dim, bias=bias)
        self.value = nn.Linear(embed_dim, head_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(K.size(-1))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, V)
        return attn_output, attn_probs

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = config["embed_dim"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = self.embed // self.num_heads
        self.qkv_bias = config["qkv_bias"]
        self.heads = nn.ModuleList([
            AttentionHead(self.embed, self.head_dim, config["attention_probs_dropout_prob"], self.qkv_bias)
            for _ in range(self.num_heads)
        ])
        self.out_proj = nn.Linear(self.embed, self.embed)
        self.out_dropout = nn.Dropout(config["dropout_val"])
    def forward(self, x, output_attentions=False):
        head_outputs = [head(x) for head in self.heads]
        attn_concat = torch.cat([out for out, _ in head_outputs], dim=-1)
        attn_out = self.out_proj(attn_concat)
        attn_out = self.out_dropout(attn_out)
        if output_attentions:
            attn_probs = torch.stack([att for _, att in head_outputs], dim=1)
            return attn_out, attn_probs
        return attn_out, None

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["embed_dim"], config["hidden_dim"])
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config["hidden_dim"], config["embed_dim"])
        self.dropout = nn.Dropout(config["dropout_val"])
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config["embed_dim"])
        self.mlp = MLP(config)
        self.norm2 = nn.LayerNorm(config["embed_dim"])
    def forward(self, x, output_attentions=False):
        attn_out, attn_probs = self.attention(self.norm1(x), output_attentions=output_attentions)
        x = x + attn_out
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        if output_attentions:
            return x, attn_probs
        return x, None

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config["num_hidden_layers"])])
    def forward(self, x, output_attentions=False):
        all_attentions = []
        for block in self.blocks:
            x, attn_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attn_probs)
        if output_attentions:
            return x, all_attentions
        return x, None

class ViTForClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(config["embed_dim"], config["num_classes"])
        self.apply(self._init_weights)
    def forward(self, x, output_attentions=False):
        embed = self.embedding(x)
        enc, all_attns = self.encoder(embed, output_attentions=output_attentions)
        logits = self.classifier(enc[:, 0])
        if output_attentions:
            return logits, all_attns
        return logits, None
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0, std=self.config["initializer_range"]
            ).to(module.position_embeddings.dtype)
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0, std=self.config["initializer_range"]
            ).to(module.cls_token.dtype)

# =====================================================
# Utility Functions for Saving & Loading Experiments
# =====================================================
def save_experiment(exp_name, config, model, train_losses, test_losses, accuracies, base_dir="experiments"):
    outdir = os.path.join(base_dir, exp_name)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config, f, sort_keys=True, indent=4)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump({"train_losses": train_losses, "test_losses": test_losses, "accuracies": accuracies},
                  f, sort_keys=True, indent=4)
    save_checkpoint(exp_name, model, "final", base_dir=base_dir)

def save_checkpoint(exp_name, model, epoch, base_dir="experiments"):
    outdir = os.path.join(base_dir, exp_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f"model_{epoch}.pt")
    torch.save(model.state_dict(), cpfile)

def load_experiment(exp_name, checkpoint_name="model_final.pt", base_dir="experiments"):
    outdir = os.path.join(base_dir, exp_name)
    with open(os.path.join(outdir, "config.json"), "r") as f:
        config = json.load(f)
    with open(os.path.join(outdir, "metrics.json"), "r") as f:
        data = json.load(f)
    model = ViTForClassification(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile))
    return config, model, data["train_losses"], data["test_losses"], data["accuracies"]

# =====================================================
# TS_Trainer: Teacher-Student Trainer
# =====================================================
class TS_Trainer:
    """
    Trainer that distills soft logits and, for ViT models, attention maps.
    If a CNN model is used, only soft logit distillation is performed.
    """
    def __init__(self, teacher_model, student_model,
                 optimizer_teacher, optimizer_student,
                 loss_fn, exp_name_teacher, exp_name_student,
                 device, alpha, scheduler_teacher, scheduler_student,
                 task_type, base_dir="experiments"):
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.optimizer_teacher = optimizer_teacher
        self.optimizer_student = optimizer_student
        self.loss_fn = loss_fn
        self.exp_name_teacher = exp_name_teacher
        self.exp_name_student = exp_name_student
        self.device = device
        self.alpha = alpha
        self.scheduler_teacher = scheduler_teacher
        self.scheduler_student = scheduler_student
        self.task_type = task_type
        self.base_dir = base_dir

    def detailed_metrics_report(self, y_true, y_pred, y_score, class_names, task_type, labels=None, display_curves=False):
        if display_curves:
            print("📊 Classification Report:")
            print(classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                        target_names=class_names, zero_division=0))
            print(f"✅ Accuracy: {accuracy_score(y_true, y_pred):.4f}")
            print(f"📌 Macro Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
            print(f"📌 Macro Recall: {recall_score(y_true, y_pred, average='macro'):.4f}")
            print(f"📌 Macro F1 Score: {f1_score(y_true, y_pred, average='macro'):.4f}")

            plt.figure(figsize=(8,6))
            y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
            for i in range(len(class_names)):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                auc_roc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc_roc:.2f})")
            plt.plot([0,1],[0,1],'k--')
            plt.title("📈 ROC Curves")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.grid(True)
            plt.show()

    def plot_roc_auc(self, y_true, y_score, task_type, title):
        if task_type == "multiclass":
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            n_classes = y_score.shape[1]
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            plt.figure(figsize=(8,6))
            for i in range(n_classes):
                plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
            plt.plot([0,1],[0,1],'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(title)
            plt.legend(loc='lower right')
            plt.show()
            return np.mean(list(roc_auc.values()))
        else:
            fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8,6))
            plt.plot(fpr, tpr, label=f'Avg ROC (AUC = {roc_auc:.2f})')
            plt.plot([0,1],[0,1],'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(title)
            plt.legend(loc='lower right')
            plt.show()
            return roc_auc

    def train_epoch_teacher(self, trainloader):
        self.teacher_model.train()
        total_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(self.device), labels.to(self.device)
            if self.task_type == "multilabel":
                labels = labels.float()
            self.optimizer_teacher.zero_grad()
            if model_name == 'ViT':
                logits, _ = self.teacher_model(images, output_attentions=False)
            else:
                logits = self.teacher_model(images)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer_teacher.step()
            total_loss += loss.item() * images.size(0)
        return total_loss / len(trainloader.dataset)

    def train_epoch_student(self, trainloader):
        self.student_model.train()
        total_loss = 0
        mse_loss_fn = nn.MSELoss()
        for images, labels in trainloader:
            images, labels = images.to(self.device), labels.to(self.device)
            if self.task_type == "multilabel":
                labels = labels.float()
            self.optimizer_student.zero_grad()
            if model_name == 'ViT':
                student_logits, student_attns = self.student_model(images, output_attentions=True)
            else:
                student_logits = self.student_model(images)
                student_attns = None
            teacher_images = F.interpolate(images, size=(224,224), mode="bilinear", align_corners=False)
            with torch.no_grad():
                if model_name == 'ViT':
                    teacher_logits, teacher_attns = self.teacher_model(teacher_images, output_attentions=True)
                else:
                    teacher_logits = self.teacher_model(teacher_images)
                    teacher_attns = None
            soft_logit_loss = mse_loss_fn(student_logits, teacher_logits)
            if model_name == 'ViT':
                num_levels = min(len(teacher_attns), len(student_attns))
                loss_attn_total = 0
                for i in range(num_levels):
                    teacher_level = teacher_attns[i][:, :, 0, 1:]
                    student_level = student_attns[i][:, :, 0, 1:]
                    teacher_level = teacher_level.mean(dim=1)
                    student_level = student_level.mean(dim=1)
                    teacher_side = int(math.sqrt(teacher_level.size(-1)))
                    student_side = int(math.sqrt(student_level.size(-1)))
                    teacher_map = teacher_level.view(-1, 1, teacher_side, teacher_side)
                    student_map = student_level.view(-1, 1, student_side, student_side)
                    teacher_map_down = F.interpolate(teacher_map, size=student_map.shape[-2:], mode="bilinear", align_corners=False)
                    level_loss = mse_loss_fn(student_map, teacher_map_down)
                    loss_attn_total += level_loss
                mlcak_loss = loss_attn_total / num_levels
                weighted_loss = self.alpha * soft_logit_loss + (1 - self.alpha) * mlcak_loss
            else:
                weighted_loss = soft_logit_loss
            weighted_loss.backward()
            self.optimizer_student.step()
            total_loss += weighted_loss.item() * images.size(0)
        return total_loss / len(trainloader.dataset)

    def evaluate_teacher(self, testloader, display_curves=False):
        self.teacher_model.eval()
        y_true, y_pred, y_score = [], [], []
        total_loss = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                if self.task_type == "multilabel":
                    labels = labels.float()
                if model_name == 'ViT':
                    logits, _ = self.teacher_model(images, output_attentions=False)
                else:
                    logits = self.teacher_model(images)
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * images.size(0)
                if self.task_type == "multiclass":
                    preds = torch.argmax(logits, dim=1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
                    y_score.extend(F.softmax(logits, dim=1).cpu().numpy())
                else:
                    preds = (logits > 0).float()
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
                    y_score.extend(torch.sigmoid(logits).cpu().numpy())
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_score = np.array(y_score)
        if self.task_type == "multiclass":
            df = pd.read_csv("/kaggle/input/csv-files/train.csv")
            df = df[["class_id", "class_name"]].drop_duplicates().sort_values("class_id")
            class_names = df["class_name"].tolist()
            acc = accuracy_score(y_true, y_pred)
        else:
            class_names = ['Aortic enlargement','Atelectasis','Calcification','Cardiomegaly',
                           'Consolidation','ILD','Infiltration','Lung Opacity','Nodule/Mass',
                           'Other lesion','Pleural effusion','Pleural thickening','Pneumothorax',
                           'Pulmonary fibrosis','No finding']
            acc = accuracy_score(y_true, y_pred)
        print(f"[Teacher] Acc: {acc:.4f}")
        if display_curves:
            auc_val = self.plot_roc_auc(y_true, y_score, self.task_type, title="Teacher ROC Curve")
            print(f"[Teacher] AUC: {auc_val:.4f}")
            self.detailed_metrics_report(y_true, y_pred, y_score, class_names, self.task_type, display_curves=True)
        else:
            if self.task_type == "multiclass":
                fpr, tpr, roc_auc = dict(), dict(), dict()
                n_classes = y_score.shape[1]
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve((y_true==i).astype(int), y_score[:,i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                auc_val = np.mean(list(roc_auc.values()))
            else:
                fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
                auc_val = auc(fpr, tpr)
            print(f"[Teacher] AUC: {auc_val:.4f}")
        return acc, total_loss / len(testloader.dataset)

    def evaluate_student(self, testloader, display_curves=False):
        self.student_model.eval()
        y_true, y_pred, y_score = [], [], []
        total_loss = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                if self.task_type == "multilabel":
                    labels = labels.float()
                if model_name == 'ViT':
                    logits, _ = self.student_model(images, output_attentions=False)
                else:
                    logits = self.student_model(images)
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * images.size(0)
                if self.task_type == "multiclass":
                    preds = torch.argmax(logits, dim=1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
                    y_score.extend(F.softmax(logits, dim=1).cpu().numpy())
                else:
                    preds = (logits > 0).float()
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
                    y_score.extend(torch.sigmoid(logits).cpu().numpy())
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_score = np.array(y_score)
        if self.task_type == "multiclass":
            df = pd.read_csv("/kaggle/input/csv-files/train.csv")
            df = df[["class_id", "class_name"]].drop_duplicates().sort_values("class_id")
            class_names = df["class_name"].tolist()
            acc = accuracy_score(y_true, y_pred)
        else:
            class_names = ['Aortic enlargement','Atelectasis','Calcification','Cardiomegaly',
                           'Consolidation','ILD','Infiltration','Lung Opacity','Nodule/Mass',
                           'Other lesion','Pleural effusion','Pleural thickening','Pneumothorax',
                           'Pulmonary fibrosis','No finding']
            acc = accuracy_score(y_true, y_pred)
        print(f"[Student] Acc: {acc:.4f}")
        if display_curves:
            auc_val = self.plot_roc_auc(y_true, y_score, self.task_type, title="Student ROC Curve")
            print(f"[Student] AUC: {auc_val:.4f}")
            self.detailed_metrics_report(y_true, y_pred, y_score, class_names, self.task_type, display_curves=True)
        else:
            if self.task_type == "multiclass":
                fpr, tpr, roc_auc = dict(), dict(), dict()
                n_classes = y_score.shape[1]
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve((y_true==i).astype(int), y_score[:,i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                auc_val = np.mean(list(roc_auc.values()))
            else:
                fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
                auc_val = auc(fpr, tpr)
            print(f"[Student] AUC: {auc_val:.4f}")
        return acc, total_loss / len(testloader.dataset)

    def train_teacher(self, config, trainloader, testloader, epochs, save_model_every_n_epochs=0, display_curves=False):
        train_losses, test_losses, accuracies = [], [], []
        self.teacher_model.train()
        for epoch in range(epochs):
            train_loss = self.train_epoch_teacher(trainloader)
            if display_curves and (epoch == epochs - 1):
                acc, test_loss = self.evaluate_teacher(testloader, display_curves=True)
            else:
                acc, test_loss = self.evaluate_teacher(testloader, display_curves=False)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(acc)
            print(f"Epoch {epoch+1} [Teacher]: Train loss={train_loss:.4f}, Test loss={test_loss:.4f}, Acc={acc:.4f}")
            self.scheduler_teacher.step()
            if save_model_every_n_epochs > 0 and ((epoch+1) % save_model_every_n_epochs == 0):
                save_checkpoint(self.exp_name_teacher, self.teacher_model, f"epoch_{epoch+1}", base_dir=self.base_dir)
        save_experiment(self.exp_name_teacher, config, self.teacher_model, train_losses, test_losses, accuracies, base_dir=self.base_dir)

    def train_student(self, config, trainloader, testloader, epochs, save_model_every_n_epochs=0, display_curves=False):
        train_losses, test_losses, accuracies = [], [], []
        self.student_model.train()
        for epoch in range(epochs):
            train_loss = self.train_epoch_student(trainloader)
            if display_curves and (epoch == epochs - 1):
                acc, test_loss = self.evaluate_student(testloader, display_curves=True)
            else:
                acc, test_loss = self.evaluate_student(testloader, display_curves=False)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(acc)
            print(f"Epoch {epoch+1} [Student]: Train loss={train_loss:.4f}, Test loss={test_loss:.4f}, Acc={acc:.4f}")
            self.scheduler_student.step()
            if save_model_every_n_epochs > 0 and ((epoch+1) % save_model_every_n_epochs == 0):
                save_checkpoint(self.exp_name_student, self.student_model, f"epoch_{epoch+1}", base_dir=self.base_dir)
        save_experiment(self.exp_name_student, config, self.student_model, train_losses, test_losses, accuracies, base_dir=self.base_dir)

# =====================================================
# Chest X-Ray Dataset Class
# =====================================================
class ChestXRayDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None, multilabel=False, label_cols=None):
        self.img_dir = img_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.multilabel = multilabel
        self.label_cols = label_cols
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        try:
            if "file_path" in self.data.columns:
                img_path = self.data.iloc[idx]["file_path"]
            else:
                img_path = os.path.join(self.img_dir, str(self.data.iloc[idx, 0]) + ".png")
            image = Image.open(img_path).convert("RGB")
            if self.multilabel:
                label = self.data.loc[idx, self.label_cols].values.astype(np.float32)
            else:
                label = int(self.data.iloc[idx, 2])
            if self.transform:
                image = self.transform(image)
            return image, label
        except (UnidentifiedImageError, OSError) as e:
            print(f"Warning: Could not load image {img_path} due to {e}. Skipping...")
            new_idx = (idx + 1) % len(self.data)
            return self.__getitem__(new_idx)

# =====================================================
# Data Preparation
# =====================================================
if TASK_TYPE == "multilabel":
    diseases = ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
                'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
                'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
                'Pulmonary fibrosis', 'No finding']
    columns_to_keep = diseases.copy()
    columns_to_keep.append("image_id")
    multilabel_csv = "/kaggle/input/csv-files/train.csv"
    train_df = pd.read_csv(multilabel_csv)
    class_mapping = train_df[["class_id", "class_name"]].drop_duplicates().sort_values(by="class_id")
    disease_labels = class_mapping["class_name"].values
    print("Disease LABELS :", disease_labels)
    print("-" * 100)
    print("Converting to one-hot vector format...")
    train_df["class_id"] = train_df["class_id"].astype(str)
    one_hot_encoded = pd.get_dummies(train_df["class_id"]).astype(int)
    train_df = pd.concat([train_df, one_hot_encoded], axis=1)
    train_df = train_df.groupby("image_id").agg({
        "0": "max", "1": "max", "2": "max", "3": "max", "4": "max",
        "5": "max", "6": "max", "7": "max", "8": "max", "9": "max",
        "10": "max", "11": "max", "12": "max", "13": "max", "14": "max"
    }).reset_index()
    print("UNIQUE images:", len(train_df.image_id.unique()))
    print("TOTAL rows:", len(train_df))
    print("Individual class counts:\n", train_df.iloc[:, 1:].sum())
    print("-" * 100)
    train_df = train_df.rename(columns=dict(zip(train_df.columns[1:], disease_labels)))
    print("Renamed columns to disease labels")
    train_df = delete_columns(train_df, columns_to_keep, root_folder="train")
    train_df = remove_all_zeros(train_df, diseases, root_folder="train")
    train_df = train_df.sample(frac=FRACTION, random_state=42)
    teacher_df = train_df.copy()
    student_df = train_df.copy()
    teacher_df = add_file_path_column(teacher_df, root_folder=f'/kaggle/input/vinbigdata-resized/vinbigdata_{TEACHER_IMAGE_SIZE}x{TEACHER_IMAGE_SIZE}/train')
    student_df["file_path"] = student_df["image_id"].apply(lambda x: os.path.join(f'/kaggle/input/vinbigdata-resized/vinbigdata_{STUDENT_IMAGE_SIZE}x{STUDENT_IMAGE_SIZE}/train', f"{x}.png"))
    teacher_csv_file = "multilabel_train_teacher.csv"
    student_csv_file = "multilabel_train_student.csv"
    teacher_df.to_csv(teacher_csv_file, index=False)
    student_df.to_csv(student_csv_file, index=False)
    label_cols = diseases
    multilabel_flag = True
else:
    multilabel_flag = False
    teacher_csv_file = "/kaggle/input/csv-files/train.csv"
    orig_df = pd.read_csv(teacher_csv_file)
    orig_df = orig_df.groupby("class_id", group_keys=False).apply(lambda x: x.sample(frac=FRACTION, random_state=42))
    multiclass_subset_csv = "multiclass_forgetting_subset.csv"
    orig_df.to_csv(multiclass_subset_csv, index=False)
    teacher_csv_file = multiclass_subset_csv
    student_csv_file = teacher_csv_file
    label_cols = None

teacher_img_dir = f"/kaggle/input/vinbigdata-resized/vinbigdata_{TEACHER_IMAGE_SIZE}x{TEACHER_IMAGE_SIZE}/train"
student_img_dir = f"/kaggle/input/vinbigdata-resized/vinbigdata_{STUDENT_IMAGE_SIZE}x{STUDENT_IMAGE_SIZE}/train"

teacher_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
student_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

teacher_dataset_full = ChestXRayDataset(img_dir=teacher_img_dir, csv_file=teacher_csv_file,
                                          transform=teacher_transform, multilabel=multilabel_flag, label_cols=label_cols if multilabel_flag else None)
student_dataset_full = ChestXRayDataset(img_dir=student_img_dir, csv_file=student_csv_file,
                                          transform=student_transform, multilabel=multilabel_flag, label_cols=label_cols if multilabel_flag else None)

if not multilabel_flag:
    stratify_arr = [teacher_dataset_full[i][1] for i in range(len(teacher_dataset_full))]
else:
    stratify_arr = teacher_dataset_full.data[diseases].values[:, 0]
train_idx, val_idx = train_test_split(range(len(teacher_dataset_full)), test_size=0.2, stratify=stratify_arr, random_state=42)

teacher_train_dataset = Subset(teacher_dataset_full, train_idx)
teacher_val_dataset = Subset(teacher_dataset_full, val_idx)
student_train_dataset = Subset(student_dataset_full, train_idx)
student_val_dataset = Subset(student_dataset_full, val_idx)

teacher_train_loader = DataLoader(teacher_train_dataset, batch_size=64, shuffle=True)
teacher_val_loader = DataLoader(teacher_val_dataset, batch_size=64, shuffle=False)
student_train_loader = DataLoader(student_train_dataset, batch_size=64, shuffle=True)
student_val_loader = DataLoader(student_val_dataset, batch_size=64, shuffle=False)

if multilabel_flag:
    num_classes = len(diseases)
else:
    num_classes = len(pd.unique(teacher_dataset_full.data["class_id"]))

# =====================================================
# Compute Weighted Loss Function
# =====================================================
if TASK_TYPE == "multilabel":
    class_weights = compute_class_weights(teacher_df[diseases], "multilabel")
    loss_fn_weighted = nn.BCEWithLogitsLoss(pos_weight=class_weights)
else:
    orig_df = pd.read_csv(teacher_csv_file)
    class_weights = compute_class_weights(orig_df, "multiclass", label_column="class_id")
    loss_fn_weighted = nn.CrossEntropyLoss(weight=class_weights)

# =====================================================
# Model Instantiation, Optimizers, and Schedulers
# =====================================================
if model_name == 'ViT':
    config_teacher = {
        "patch_size": 16,
        "embed_dim": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "hidden_dim": 64 * 16,
        "dropout_val": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "image_size": TEACHER_IMAGE_SIZE,
        "num_classes": num_classes,
        "num_channels": 3,
        "qkv_bias": True,
        "use_faster_attention": True,
        "attention_block_index": 1,
        "task_type": TASK_TYPE
    }
    config_student = {
        "patch_size": 4,
        "embed_dim": 64,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "hidden_dim": 64 * 8,
        "dropout_val": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "image_size": STUDENT_IMAGE_SIZE,
        "num_classes": num_classes,
        "num_channels": 3,
        "qkv_bias": True,
        "use_faster_attention": True,
        "attention_block_index": 1,
        "task_type": TASK_TYPE
    }
    teacher_model = ViTForClassification(config_teacher).to(device)
    student_model = ViTForClassification(config_student).to(device)
else:
    teacher_model = {
        'vgg19': models.vgg19(weights=None, num_classes=num_classes),
        'resnet18': models.resnet18(weights=None, num_classes=num_classes),
        'resnet34': models.resnet34(weights=None, num_classes=num_classes),
        'resnet50': models.resnet50(weights=None, num_classes=num_classes)
    }[model_name].to(device)
    student_model = {
        'vgg19': models.vgg19(weights=None, num_classes=num_classes),
        'resnet18': models.resnet18(weights=None, num_classes=num_classes),
        'resnet34': models.resnet34(weights=None, num_classes=num_classes),
        'resnet50': models.resnet50(weights=None, num_classes=num_classes)
    }[model_name].to(device)

optimizer_teacher = optim.AdamW(teacher_model.parameters(), lr=5e-4)
optimizer_student = optim.AdamW(student_model.parameters(), lr=5e-4)
loss_fn = loss_fn_weighted
epoch_teacher = 100
epoch_student = 100
scheduler_teacher = CosineAnnealingLR(optimizer_teacher, T_max=epoch_teacher)
scheduler_student = CosineAnnealingLR(optimizer_student, T_max=epoch_student)

exp_name_teacher = f"Model_Teacher_{model_name}_{TEACHER_IMAGE_SIZE}_{TASK_TYPE}_{FRACTION}_ChestXRay"
exp_name_student = f"Model_Student_{model_name}_{STUDENT_IMAGE_SIZE}_{TASK_TYPE}_{FRACTION}_ChestXRay"

# =====================================================
# Instantiate TS_Trainer
# =====================================================
trainer = TS_Trainer(teacher_model, student_model,
                     optimizer_teacher, optimizer_student,
                     loss_fn, exp_name_teacher, exp_name_student,
                     device, alpha=0.5, scheduler_teacher=scheduler_teacher,
                     scheduler_student=scheduler_student, task_type=TASK_TYPE, base_dir="experiments")

# =====================================================
# Forgetting Tracker Initialization
# =====================================================
num_samples = len(teacher_train_dataset)
forgetting_tracker = initialize_forgetting_tracker(num_samples)

def update_forgetting_during_training(model, train_loader, epoch, tracker):
    model.eval()
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            if model_name == 'ViT':
                outputs, _ = model(images)
            else:
                outputs = model(images)
            if TASK_TYPE == "multiclass":
                predicted = torch.argmax(outputs, dim=1)
            else:
                predicted = (outputs > 0).float()
            update_forgetting_tracker(tracker, labels.cpu().numpy(), predicted.cpu().numpy(), epoch, task_type=TASK_TYPE)

# =====================================================
# Training: Teacher and Student
# =====================================================
print(f"Training {model_name} Teacher Model with Weighted Loss & Forgetting Updates...")
for epoch in range(epoch_teacher):
    trainer.train_teacher(config_teacher if model_name=='ViT' else None,
                          teacher_train_loader, teacher_val_loader,
                          1, save_model_every_n_epochs=5, display_curves=(epoch == epoch_teacher - 1))
    # Only update forgetting tracker if FRACTION is NOT 1.0
    if FRACTION != 1.0:
        update_forgetting_during_training(teacher_model, teacher_train_loader, epoch, forgetting_tracker)

print(f"Training {model_name} Student Model with Weighted Loss & Forgetting Updates...")
for epoch in range(epoch_student):
    trainer.train_student(config_student if model_name=='ViT' else None,
                          student_train_loader, student_val_loader,
                          1, save_model_every_n_epochs=5, display_curves=(epoch == epoch_student - 1))
    if FRACTION != 1.0:
        update_forgetting_during_training(student_model, student_train_loader, epoch, forgetting_tracker)

# =====================================================
# Attention Visualization (Only for ViT)
# =====================================================
def TS_visualize_attention_chest(teacher_model, student_model, dataset, indices=None, output=None, device="cuda"):
    if model_name != 'ViT':
        print("Attention visualization is only available for the ViT model.")
        return
    teacher_model.eval()
    student_model.eval()
    if indices is None:
        num_images = 5
        indices = torch.randperm(len(dataset))[:num_images]
    else:
        num_images = len(indices)
    raw_images = []
    image_tensors = []
    for idx in indices:
        img, _ = dataset[idx]
        image_tensors.append(img)
        img_np = img.cpu().numpy().transpose(1,2,0)
        mean = np.array([0.485,0.456,0.406])
        std = np.array([0.229,0.224,0.225])
        img_denorm = std * img_np + mean
        img_denorm = np.clip(img_denorm, 0, 1)
        raw_images.append(img_denorm)
    images = torch.stack(image_tensors).to(device)
    with torch.no_grad():
        teacher_logits, teacher_attns = teacher_model(F.interpolate(images, size=(224,224), mode="bilinear", align_corners=False), output_attentions=True)
        student_logits, student_attns = student_model(images, output_attentions=True)
    if TASK_TYPE == "multiclass":
        teacher_preds = torch.argmax(teacher_logits, dim=1)
        student_preds = torch.argmax(student_logits, dim=1)
        mapping_df = pd.read_csv("/kaggle/input/csv-files/train.csv")
        mapping_df = mapping_df[["class_id", "class_name"]].drop_duplicates().sort_values("class_id")
        class_mapping = {int(row["class_id"]): row["class_name"] for _, row in mapping_df.iterrows()}
        teacher_pred_names = [class_mapping.get(int(pred.item()), str(pred.item())) for pred in teacher_preds]
        student_pred_names = [class_mapping.get(int(pred.item()), str(pred.item())) for pred in student_preds]
    else:
        teacher_pred_vec = (teacher_logits > 0).float().cpu().numpy()
        student_pred_vec = (student_logits > 0).float().cpu().numpy()
        teacher_pred_names = []
        student_pred_names = []
        for pred in teacher_pred_vec:
            names = [d for d, v in zip(diseases, pred) if v == 1]
            teacher_pred_names.append(", ".join(names) if names else "None")
        for pred in student_pred_vec:
            names = [d for d, v in zip(diseases, pred) if v == 1]
            student_pred_names.append(", ".join(names) if names else "None")
    teacher_cls_attn = teacher_attns[0][:, :, 0, 1:].mean(dim=1)
    student_cls_attn = student_attns[0][:, :, 0, 1:].mean(dim=1)
    teacher_side = int(math.sqrt(teacher_cls_attn.size(-1)))
    student_side = int(math.sqrt(student_cls_attn.size(-1)))
    teacher_map = teacher_cls_attn.view(-1, 1, teacher_side, teacher_side)
    student_map = student_cls_attn.view(-1, 1, student_side, student_side)
    teacher_map_down = F.interpolate(teacher_map, size=student_map.shape[-2:], mode="bilinear", align_corners=False)
    fig, axes = plt.subplots(num_images, 3, figsize=(15,8))
    for i in range(num_images):
        axes[i,0].imshow(raw_images[i])
        axes[i,0].axis("off")
        axes[i,0].set_title(f"Image {i+1}")
        axes[i,1].imshow(teacher_map_down[i].detach().cpu().squeeze(), cmap="jet")
        axes[i,1].axis("off")
        axes[i,1].set_title(f"Teacher Attn\nPred: {teacher_pred_names[i]}")
        axes[i,2].imshow(student_map[i].detach().cpu().squeeze(), cmap="jet")
        axes[i,2].axis("off")
        axes[i,2].set_title(f"Student Attn\nPred: {student_pred_names[i]}")
    plt.tight_layout()
    if output is not None:
        plt.savefig(output)
    plt.show()

# Visualize attention maps only for ViT models
TS_visualize_attention_chest(teacher_model, student_model, student_val_dataset, device=device)
