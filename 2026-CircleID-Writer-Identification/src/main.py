# --- MAIN EXECUTION CELL ---
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 1. Load Data
train_df = pd.read_csv(TRAIN_CSV)
le = LabelEncoder()
train_df['label'] = le.fit_transform(train_df['writer_id'])

# 2. Setup Data
train_subset, val_subset = train_test_split(train_df, test_size=0.1, stratify=train_df['label'])
train_loader = DataLoader(CircleDataset(train_subset, IMAGE_ROOT, get_transforms(IMG_SIZE, 'train')), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(CircleDataset(val_subset, IMAGE_ROOT, get_transforms(IMG_SIZE, 'val')), batch_size=BATCH_SIZE)

# 3. Model, Loss, Optimizer
model = get_model(num_classes=len(le.classes_)).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4. Run Training
for epoch in range(EPOCHS):
    loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}")

# 5. Inference & Save
test_df = pd.read_csv(TEST_CSV)
test_loader = DataLoader(CircleDataset(test_df, IMAGE_ROOT, get_transforms(IMG_SIZE, 'val'), is_test=True), batch_size=BATCH_SIZE)
submission = run_inference(model, test_loader, le, CONFIDENCE_THRESHOLD, DEVICE)
submission.to_csv("submission_writer.csv", index=False)
