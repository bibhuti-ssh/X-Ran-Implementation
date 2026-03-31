"""
Generate presentation slides for XRan project evaluation.
Creates a visual summary using matplotlib for the demo/presentation.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def create_presentation_slides(output_dir='presentation'):
    os.makedirs(output_dir, exist_ok=True)

    # ===== Slide 1: Title =====
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor('#1a1a2e')

    ax.text(0.5, 0.65, 'XRan', fontsize=72, fontweight='bold',
            ha='center', va='center', color='#e94560')
    ax.text(0.5, 0.48, 'Explainable Deep Learning-based\nRansomware Detection',
            fontsize=28, ha='center', va='center', color='white')
    ax.text(0.5, 0.30, 'Using Dynamic Analysis with Windows API Calls',
            fontsize=20, ha='center', va='center', color='#a0a0a0')
    ax.text(0.5, 0.15, 'CSEC Assignment - Paper Implementation',
            fontsize=16, ha='center', va='center', color='#a0a0a0')
    ax.text(0.5, 0.05, 'Gulmez et al., Computers & Security, 2024',
            fontsize=14, ha='center', va='center', color='#606060',
            fontstyle='italic')

    plt.savefig(os.path.join(output_dir, 'slide_01_title.png'),
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    # ===== Slide 2: Problem Statement =====
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor('#f5f5f5')

    ax.text(0.5, 0.92, 'Problem Statement', fontsize=36, fontweight='bold',
            ha='center', va='center', color='#1a1a2e')

    problems = [
        ('Ransomware Threat', 'Attacks increased 126% in 2023; billions lost annually'),
        ('Signature Limitations', 'Traditional detection fails against new/obfuscated variants'),
        ('Single-Feature Analysis', 'Existing methods use only API calls or frequency features'),
        ('Black-Box Models', 'Deep learning models provide no decision explainability'),
    ]

    for i, (title, desc) in enumerate(problems):
        y = 0.75 - i * 0.18
        ax.add_patch(mpatches.FancyBboxPatch((0.05, y - 0.06), 0.9, 0.14,
                     boxstyle="round,pad=0.01", facecolor='white',
                     edgecolor='#e94560', linewidth=2))
        ax.text(0.10, y + 0.01, f'{i+1}. {title}', fontsize=18,
                fontweight='bold', va='center', color='#1a1a2e')
        ax.text(0.10, y - 0.03, desc, fontsize=14, va='center', color='#404040')

    plt.savefig(os.path.join(output_dir, 'slide_02_problem.png'),
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    # ===== Slide 3: XRan Architecture =====
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor('#f5f5f5')

    ax.text(0.5, 0.95, 'XRan Architecture', fontsize=36, fontweight='bold',
            ha='center', va='center', color='#1a1a2e')

    # Pipeline boxes
    stages = [
        ('Cuckoo\nSandbox', '#4472C4', 0.08),
        ('API Calls\n(500)', '#ED7D31', 0.24),
        ('DLLs\n(10)', '#ED7D31', 0.38),
        ('Mutexes\n(10)', '#ED7D31', 0.52),
        ('2-Layer\nCNN', '#70AD47', 0.68),
        ('Ransomware\n/ Benign', '#e94560', 0.84),
    ]

    for label, color, x in stages:
        ax.add_patch(mpatches.FancyBboxPatch((x, 0.55), 0.12, 0.2,
                     boxstyle="round,pad=0.01", facecolor=color,
                     edgecolor='white', linewidth=2, alpha=0.9))
        ax.text(x + 0.06, 0.65, label, fontsize=12, fontweight='bold',
                ha='center', va='center', color='white')

    # Arrows
    for i in range(len(stages) - 1):
        ax.annotate('', xy=(stages[i+1][2], 0.65),
                    xytext=(stages[i][2] + 0.12, 0.65),
                    arrowprops=dict(arrowstyle='->', color='#404040', lw=2))

    # Bottom details
    details = [
        'Combined Sequence: A_500 || D_10 || M_10 = 520 features',
        'Conv1D(128,k=5) -> ReLU -> MaxPool -> Dropout -> Conv1D(64,k=3) -> ReLU -> MaxPool -> Dropout -> Dense -> Sigmoid',
        'Loss: Binary CrossEntropy | Optimizer: Adam | Epochs: 10',
        'XAI: LIME (local explanations) + SHAP (global explanations)',
    ]
    for i, detail in enumerate(details):
        ax.text(0.5, 0.40 - i * 0.08, detail, fontsize=13,
                ha='center', va='center', color='#1a1a2e')

    plt.savefig(os.path.join(output_dir, 'slide_03_architecture.png'),
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    # ===== Slide 4: Results Table =====
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor('#f5f5f5')

    ax.text(0.5, 0.95, 'Experimental Results', fontsize=36, fontweight='bold',
            ha='center', va='center', color='#1a1a2e')

    # Table data
    headers = ['Model', 'Accuracy', 'TPR', 'FPR', 'F1-Score']
    data = [
        ['Decision Tree', '0.908', '0.870', '0.067', '0.883'],
        ['Random Forest', '0.964', '0.910', '0.000', '0.953'],
        ['Naive Bayes', '0.484', '0.820', '0.740', '0.560'],
        ['KNN (k=3)', '0.544', '0.920', '0.707', '0.617'],
        ['KNN (k=5)', '0.548', '0.960', '0.727', '0.630'],
        ['LSTM', '1.000', '1.000', '0.000', '1.000'],
        ['CNN (1-layer)', '1.000', '1.000', '0.000', '1.000'],
        ['XRan (2L-CNN)', '1.000', '1.000', '0.000', '1.000'],
    ]

    table = ax.table(cellText=data, colLabels=headers,
                     loc='center', cellLoc='center',
                     bbox=[0.05, 0.05, 0.9, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(14)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor('#1a1a2e')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Highlight XRan row
    for j in range(len(headers)):
        table[len(data), j].set_facecolor('#e94560')
        table[len(data), j].set_text_props(color='white', fontweight='bold')

    plt.savefig(os.path.join(output_dir, 'slide_04_results.png'),
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    # ===== Slide 5: Model Comparison Chart =====
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor('#f5f5f5')
    fig.suptitle('Performance Comparison', fontsize=28, fontweight='bold',
                 color='#1a1a2e', y=0.98)

    models = ['DT', 'RF', 'NB', 'KNN-3', 'KNN-5', 'LSTM', 'CNN', 'XRan']
    acc = [0.908, 0.964, 0.484, 0.544, 0.548, 1.0, 1.0, 1.0]
    tpr = [0.870, 0.910, 0.820, 0.920, 0.960, 1.0, 1.0, 1.0]
    fpr = [0.067, 0.000, 0.740, 0.707, 0.727, 0.000, 0.000, 0.000]
    f1 = [0.883, 0.953, 0.560, 0.617, 0.630, 1.0, 1.0, 1.0]

    colors = ['#4472C4'] * 5 + ['#ED7D31'] * 2 + ['#e94560']

    axes[0].bar(models, acc, color=colors)
    axes[0].set_title('Accuracy', fontsize=16, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(acc):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)

    axes[1].bar(models, f1, color=colors)
    axes[1].set_title('F1-Score', fontsize=16, fontweight='bold')
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(f1):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)

    legend_elements = [
        mpatches.Patch(facecolor='#4472C4', label='ML Baselines'),
        mpatches.Patch(facecolor='#ED7D31', label='DL Baselines'),
        mpatches.Patch(facecolor='#e94560', label='XRan (Proposed)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=14)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'slide_05_comparison.png'),
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    # ===== Slide 6: XAI Explanations =====
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor('#f5f5f5')

    ax.text(0.5, 0.95, 'Explainability (XAI)', fontsize=36, fontweight='bold',
            ha='center', va='center', color='#1a1a2e')

    # LIME section
    ax.add_patch(mpatches.FancyBboxPatch((0.02, 0.45), 0.45, 0.40,
                 boxstyle="round,pad=0.02", facecolor='white',
                 edgecolor='#4472C4', linewidth=3))
    ax.text(0.245, 0.82, 'LIME (Local)', fontsize=22, fontweight='bold',
            ha='center', color='#4472C4')
    lime_text = (
        'Per-prediction explanations\n\n'
        'Ransomware indicators:\n'
        '  - NtAllocateVirtualMemory\n'
        '  - RegOpenKeyExW\n'
        '  - Advapi32.dll\n'
        '  - ZonesCounterMutex'
    )
    ax.text(0.245, 0.62, lime_text, fontsize=13, ha='center', va='center',
            color='#1a1a2e', family='monospace')

    # SHAP section
    ax.add_patch(mpatches.FancyBboxPatch((0.53, 0.45), 0.45, 0.40,
                 boxstyle="round,pad=0.02", facecolor='white',
                 edgecolor='#ED7D31', linewidth=3))
    ax.text(0.755, 0.82, 'SHAP (Global)', fontsize=22, fontweight='bold',
            ha='center', color='#ED7D31')
    shap_text = (
        'Global feature importance\n\n'
        'Key finding from paper:\n'
        '  DLLs and Mutexes (20 positions)\n'
        '  are MORE important than\n'
        '  API calls (500 positions)'
    )
    ax.text(0.755, 0.62, shap_text, fontsize=13, ha='center', va='center',
            color='#1a1a2e', family='monospace')

    # Bottom note
    ax.text(0.5, 0.30, 'Key Insight: Multi-view feature fusion captures cross-feature correlations',
            fontsize=16, ha='center', va='center', color='#e94560', fontweight='bold')
    ax.text(0.5, 0.18,
            'Combining API calls + DLLs + Mutexes in sequence form preserves ordering relationships\n'
            'that are lost in frequency-based approaches used by prior work.',
            fontsize=13, ha='center', va='center', color='#404040')

    plt.savefig(os.path.join(output_dir, 'slide_06_xai.png'),
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    # ===== Slide 7: Conclusions =====
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor('#1a1a2e')

    ax.text(0.5, 0.90, 'Conclusions', fontsize=36, fontweight='bold',
            ha='center', va='center', color='white')

    conclusions = [
        'XRan achieves up to 99.4% TPR, outperforming all state-of-the-art methods',
        'Multi-view feature combination (API + DLL + Mutex) enriches the feature space',
        '2-layer CNN captures hierarchical patterns in the combined sequences',
        'DLLs and Mutexes provide stronger discriminative signals than API calls alone',
        'LIME and SHAP provide actionable insights for security analysts',
        'Dynamic analysis ensures resilience against code obfuscation/encryption',
    ]

    for i, conclusion in enumerate(conclusions):
        y = 0.75 - i * 0.10
        ax.text(0.08, y, f'   {conclusion}', fontsize=16,
                va='center', color='white')
        ax.plot(0.05, y, 'o', color='#e94560', markersize=10)

    ax.text(0.5, 0.08, 'Thank You', fontsize=28, fontweight='bold',
            ha='center', va='center', color='#e94560')

    plt.savefig(os.path.join(output_dir, 'slide_07_conclusions.png'),
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    print(f"Presentation slides saved to {output_dir}/")
    print("Generated 7 slides:")
    for i, name in enumerate([
        'Title', 'Problem Statement', 'Architecture',
        'Results Table', 'Performance Comparison',
        'XAI Explanations', 'Conclusions'
    ]):
        print(f"  Slide {i+1}: {name}")


if __name__ == '__main__':
    create_presentation_slides(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presentation')
    )
