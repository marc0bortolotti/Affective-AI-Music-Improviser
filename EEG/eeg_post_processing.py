from matplotlib import pyplot as plt
import numpy as np

N_USERS = 3

accuracies = []

for user_id in range(N_USERS):
    for emotion in ['RELAXED', 'EXCITED']: 
        path = f"C:/Users/Gianni/Desktop/MARCO/UNI/Magistrale/TESI/User Study/experiments/user_{user_id}/test_BCI_{emotion}_0/emotions.txt"

        with open(path, 'r') as f:
            lines = f.readlines()
            n_lines = len(lines)    
        
        n_correct = 0
        n_wrong = 0
        for line in lines:
            if emotion == 'RELAXED' and 'R' in line:
                n_correct += 1
            elif emotion == 'EXCITED' and 'C' in line:
                n_correct += 1

        accuracy = n_correct / n_lines
        accuracies.append(accuracy)

print(accuracies)

# plot accuracies divided by user and emotion 
n = 2
relaxed = accuracies[::n]
excited = accuracies[1::n]
fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(N_USERS)
bar1 = ax.bar(index, relaxed, bar_width, label='Relaxed', color='skyblue')
bar2 = ax.bar(index + bar_width, excited, bar_width, label='Excited', color='coral')

# add the accuracy values on top of the bars

for i, v in enumerate(relaxed):
    ax.text(i - 0.1, v + 0.01, str(round(v, 2)), color='grey', fontweight='bold')

for i, v in enumerate(excited):
    ax.text(i + 0.25, v + 0.01, str(round(v, 2)), color='grey', fontweight='bold')

ax.set_xlabel('User')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by user and emotion')
ax.set_xticks(index + bar_width / 2)
labels = []
for i in range(N_USERS):
    labels.append(f'User {i}')
ax.set_xticklabels(labels)
ax.legend()
plt.show()

# save plot
fig.savefig('eeg_classifier_accuracies.png')

