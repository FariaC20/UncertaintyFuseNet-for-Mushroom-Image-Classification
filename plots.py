import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from scipy.interpolate import interp1d
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

matplotlib.rc('xtick', labelsize=22)
matplotlib.rc('ytick', labelsize=22)
plt.rcParams.update({'font.size': 22})


from scipy.interpolate import interp1d

def plot_roc_handy(y_test, y_score, lw=2, name='Roc', class_name=None, zoom=False, axis=None):
    if axis is None:
        axis = [0.0, 0.12, 0.88, 1.0]
    if class_name is None:
        class_name = ['Lactarius', 'Amanita', 'Boletus', 'Russula', 'Entoloma', 'Hygrocybe', 'Cortinarius', 'Suillus', 'Agaricus']
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute ROC curve and ROC area for each class
    for i in range(y_test.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Aggregate all FPRs
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(y_test.shape[1])]))
    mean_tpr = np.zeros_like(all_fpr)

    # Interpolate all TPRs
    for i in range(y_test.shape[1]):
        interpolation = interp1d(fpr[i], tpr[i], kind="linear", bounds_error=False, fill_value=0)
        mean_tpr += interpolation(all_fpr)

    # Average TPR and compute macro-average ROC
    mean_tpr /= y_test.shape[1]
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plotting
    f, ax = plt.subplots(figsize=[15, 15])
    ax.plot(fpr["micro"], tpr["micro"],
            label=f'Micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
            color='deeppink', linestyle=':', linewidth=4)
    ax.plot(fpr["macro"], tpr["macro"],
            label=f'Macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'cyan', 'magenta'])
    for i, color in zip(range(y_test.shape[1]), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                label=f'ROC curve of class {class_name[i]} (area = {roc_auc[i]:0.2f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=22)
    ax.set_ylabel('True Positive Rate', fontsize=22)
    ax.set_title(name, fontsize=22)
    ax.legend(loc="lower right")

    # Optional zoomed inset
    if zoom:
        axins = ax.inset_axes([0.3, 0.4, 0.4, 0.4])
        for i, color in zip(range(y_test.shape[1]), colors):
            axins.plot(fpr[i], tpr[i], color=color, lw=lw)
        axins.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4)
        axins.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', linewidth=4)
        x1, x2, y1, y2 = axis
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticklabels('')
        axins.set_yticklabels('')
        ax.indicate_inset_zoom(axins)

    plt.show()
    f.savefig(f'{name}.pdf', bbox_inches='tight')
    ax.figure.savefig("{}.pdf".format(name), bbox_inches='tight')


def plot_cm_handy(y_test, y_score, lw=2, name='Confusion Matrix of Fusion Model without Uncertainty (Mushroom_Image)',
                  class_name=None):

    if class_name is None:
        class_name = ['Lactarius', 'Amanita', 'Boletus', 'Russula', 'Entoloma', 'Hygrocybe', 'Cortinarius', 'Suillus', 'Agaricus']
    CM = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_score, axis=1))
    cm = CM
    cmap = plt.cm.Blues
    fig, ax = plt.subplots(figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(name)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_name, yticklabels=class_name,
           ylabel='True Label',
           xlabel='Predicted Label')
    ax.set_xticklabels(class_name, fontsize=15)
    ax.set_yticklabels(class_name, fontsize=15)
    ax.set_ylabel('True Label', fontsize=15)
    ax.set_xlabel('Predicted Label', fontsize=15)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fmt = '.1f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            figure(num=None, figsize=(5, 5), dpi=80, facecolor='w', edgecolor='k')
            ax.text(j, i, format(cm[i, j], fmt), fontsize=12,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig('{}.pdf'.format(name), dpi=300)
    ax.figure.savefig("{}.pdf".format(name), bbox_inches='tight')


def uncertainty_plot(model, image, label=None, save=True, name=None, mc_iter=200, dataset='Mushroom_Image', ylim=20):
    image = image[np.newaxis, :, :, :]
    if label is not None:
        label_idx = np.argmax(label)

    if dataset == 'Mushroom_Image':
        class_text = ['Lactarius', 'Amanita', 'Boletus', 'Russula', 'Entoloma', 'Hygrocybe', 'Cortinarius', 'Suillus', 'Agaricus']
    #elif dataset == 'CT':
        #class_text = ['nCT', 'NiCT', 'pCT']

    all_preds = []
    for _ in range(mc_iter):
        preds = model.predict(image)
        all_preds.append(preds)

    preds_mean = np.mean(all_preds, axis=0)
    preds_mean = preds_mean[0]
    preds_std = np.std(all_preds, axis=0)
    preds_std = preds_std[0]
    class_idx_mean = np.argmax(preds_mean)

    preds = np.transpose(np.stack(all_preds), (1, 0, 2))

    image = (image + 1) / 2
    image1 = (image * 255).astype("uint8")
    image1 = np.squeeze(image1, axis=0)
    image1 = np.squeeze(image1, axis=2)

    fig = plt.figure(figsize=[8, 6.5])

    plt.subplot(2, 1, 1)
    plt.imshow(image1, cmap='gray')
    if label is not None:
        title = """
    True: {}
    Prediction: {}
    Value Order: [{}, {}, {}]
    Mean: [{:0.2f}, {:0.2f}, {:0.2f}]
    STD: [{:0.2f}, {:0.2f}, {:0.2f}]
    """.format(class_text[label_idx], class_text[class_idx_mean], *class_text,
               *preds_mean, *preds_std)
    else:
        title = """
    Prediction: {}
    Value Order: [{}, {}, {}]
    Mean: [{:0.2f}, {:0.2f}, {:0.2f}]
    STD: [{:0.2f}, {:0.2f}, {:0.2f}]
    """.format(class_text[class_idx_mean], *class_text,
               *preds_mean, *preds_std)

    plt.title(title, fontsize=12)

    plt.axis('off')

    for i in range(3):
        plt.subplot(2, 1, 2)
        plt.hist(preds[0][:, i], bins=50, alpha=0.3, label=class_text[i])

        plt.axvline(np.quantile(preds[0][:, i], 0.5), color='red', linestyle='--', alpha=0.4)
        plt.axvline(0.5, color='green', linestyle='--')

        plt.xlabel('Uncertainty', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.ylim([0, ylim])
        plt.legend()
    plt.tight_layout()
    if save:
        fig.savefig('{}.pdf'.format(name), dpi=300, bbox_inches='tight')
    plt.show()
