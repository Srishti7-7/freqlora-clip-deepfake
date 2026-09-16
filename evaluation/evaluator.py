"""
evaluation/evaluator.py - Evaluation on test set
"""

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, \
                           precision_recall_fscore_support

class Evaluator:
    """Evaluate model on test set"""
    
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device
    
    def evaluate(self, test_loader):
        """Evaluate and return metrics"""
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device)
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = logits.argmax(dim=1)
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.numpy())
                all_probs.append(probs.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        
        # Compute metrics
        acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        return {
            'accuracy': acc,
            'auc': auc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        }
