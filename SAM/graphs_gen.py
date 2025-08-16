import matplotlib.pyplot as plt

# Read values from text files
def read_losses(file_path):
    with open(file_path, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

# Replace with your actual file names
train_loss = read_losses('Meas_merged_2\iou_arr_train_0.txt')
val_loss = read_losses('Meas_merged_2\iou_arr_val_0.txt')

plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Train IOU')
plt.plot(val_loss, label='Validation IOU')
plt.xlabel('Epoch')
plt.ylabel('IOU')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('iou_plot.png')  # Save figure
plt.show()
