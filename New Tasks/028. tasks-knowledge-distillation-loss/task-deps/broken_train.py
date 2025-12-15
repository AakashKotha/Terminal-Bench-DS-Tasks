import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from models import TeacherModel, StudentModel

DATA_PATH = "/app/data/dataset.pt"
OUTPUT_PATH = "/app/output/student.pth"

def load_data():
    data = torch.load(DATA_PATH)
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]

def get_pretrained_teacher(X, y):
    # Simulate loading a pre-trained teacher by quickly training one on the spot
    # Ideally this would be loaded from disk, but for self-containment we train it here.
    # The teacher is "Smart" so it learns fast.
    print("Setting up Teacher Model...")
    teacher = TeacherModel()
    opt = optim.Adam(teacher.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    
    for _ in range(50): # Quick training
        opt.zero_grad()
        loss = crit(teacher(X), y)
        loss.backward()
        opt.step()
    
    teacher.eval()
    acc = (teacher(X).argmax(1) == y).float().mean()
    print(f"Teacher Train Accuracy: {acc:.2%}")
    return teacher

def compute_distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    """
    Calculates KD Loss:
    Loss = alpha * KL(Student, Teacher) + (1-alpha) * CE(Student, Labels)
    """
    
    # --- BROKEN LOGIC START ---
    
    # ERROR 1: PyTorch nn.KLDivLoss expects input to be Log-Probabilities (using log_softmax).
    # Here we are just applying softmax (Probabilities). 
    # This fundamentally breaks the math of KL divergence calculation in PyTorch.
    student_soft = F.softmax(student_logits / T, dim=1)
    
    # ERROR 2: PyTorch nn.KLDivLoss expects target to be Probabilities (which is correct here),
    # BUT, since we used softmax on input above (Error 1), the gradients will be wrong.
    # Correct implementation: Input=log_softmax, Target=softmax.
    teacher_soft = F.softmax(teacher_logits / T, dim=1)

    kl_crit = nn.KLDivLoss(reduction="batchmean")
    
    # ERROR 3: Missing T^2 scaling.
    # Since we divided logits by T, the gradients are scaled down by 1/T^2.
    # We must multiply the distillation loss by T^2 to keep it comparable to the CE loss.
    # Without this, the soft-target signal is tiny compared to the hard label signal.
    
    distillation_loss = kl_crit(student_soft, teacher_soft)
    
    # --- BROKEN LOGIC END ---

    student_loss = F.cross_entropy(student_logits, labels)
    
    total_loss = (alpha * distillation_loss) + ((1 - alpha) * student_loss)
    return total_loss

def train_student():
    X_train, y_train, X_test, y_test = load_data()
    teacher = get_pretrained_teacher(X_train, y_train)
    
    print("\nTraining Student with Distillation...")
    student = StudentModel()
    optimizer = optim.Adam(student.parameters(), lr=0.01)
    
    # Train Loop
    epochs = 200
    for epoch in range(epochs):
        student.train()
        optimizer.zero_grad()
        
        # Forward
        s_logits = student(X_train)
        
        with torch.no_grad():
            t_logits = teacher(X_train)
            
        loss = compute_distillation_loss(s_logits, t_logits, y_train, T=4.0, alpha=0.7)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            # Eval
            student.eval()
            with torch.no_grad():
                test_preds = student(X_test).argmax(1)
                acc = (test_preds == y_test).float().mean()
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Test Acc={acc:.2%}")

    # Final Eval
    student.eval()
    final_acc = (student(X_test).argmax(1) == y_test).float().mean().item()
    print(f"\nFinal Student Accuracy: {final_acc:.2%}")
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(student.state_dict(), OUTPUT_PATH)
    
    # Threshold check for self-diagnosis
    if final_acc < 0.82:
        print("FAILURE: Student accuracy is too low. Distillation logic is likely flawed.")
    else:
        print("SUCCESS: Student successfully distilled knowledge.")

if __name__ == "__main__":
    train_student()