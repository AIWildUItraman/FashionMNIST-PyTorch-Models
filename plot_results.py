import matplotlib.pyplot as plt

epochs = range(1, 21)
train_acc = [0.884, 0.904, 0.917, 0.927, 0.932, 0.937, 0.945, 0.949, 0.953, 0.958, 0.962, 0.967, 0.971, 0.974, 0.976, 0.979, 0.981, 0.984, 0.984, 0.987]
test_acc = [0.869, 0.841, 0.908, 0.913, 0.913, 0.905, 0.910, 0.913, 0.909, 0.902, 0.921, 0.920, 0.905, 0.916, 0.917, 0.919, 0.917, 0.920, 0.921, 0.921]
plt.figure(figsize=(12, 8)) # 设置图像大小为 12*8
plt.plot(epochs, train_acc, '-o', label='train_acc')
plt.plot(epochs, test_acc, '-o', label='test_acc')
plt.xticks(range(1, len(train_acc) + 1))
best_epoch = test_acc.index(max(test_acc)) + 1
best_acc = max(test_acc)
plt.annotate(f'best epoch: {best_epoch}\nbest test_acc: {best_acc:.3f}', xy=(best_epoch, best_acc), xytext=(best_epoch, best_acc + 0.01), ha='center', va='bottom', arrowprops=dict(arrowstyle='->'))

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy')
plt.legend()
plt.savefig('/home/mxs/桌面/FashionMNIST-PyTorch-Models/imges/results.png')
