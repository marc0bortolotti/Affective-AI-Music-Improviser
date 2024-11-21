from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

N_USERS = 3

accuracies = []
variances = []  

for user_id in range(N_USERS):
    for emotion in ['RELAXED', 'EXCITED']: 
        folder_path = f"C:/Users/Gianni/Desktop/MARCO/UNI/Magistrale/TESI/User Study/experiments/user_{user_id}/test_BCI_{emotion}_0"

        path = f"{folder_path}/emotions.txt"
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

        path = f"{folder_path}/session.csv"
        data = pd.read_csv(filepath_or_buffer=path, delimiter='\t')
        acc = data.iloc[:, 8:11].to_numpy(dtype=np.float64)
        gyro = data.iloc[:, 11:14].to_numpy(dtype=np.float64)

        # get the variance of accelerometer data
        acc_var = np.var(acc, axis=0)
        # gyro_var = np.var(gyro, axis=0)
        variances.append(acc_var)

print(variances)
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
# plt.show()

# # save plot
# fig.savefig('eeg_classifier_accuracies.png')


# plot variances 
n = 2
relaxed = variances[::n]
excited = variances[1::n]
fig, ax = plt.subplots()
bar_width = 0.1
index = np.arange(N_USERS)

relaxed_x = [relaxed[i][0] for i in range(N_USERS)]
relaxed_y = [relaxed[i][1] for i in range(N_USERS)]
relaxed_z = [relaxed[i][2] for i in range(N_USERS)]
excited_x = [excited[i][0] for i in range(N_USERS)]
excited_y = [excited[i][1] for i in range(N_USERS)]
excited_z = [excited[i][2] for i in range(N_USERS)]

bar1 = ax.bar(index - 3*bar_width, relaxed_x, bar_width, label='Relaxed', tick_label = 'x', color='skyblue')
bar2 = ax.bar(index - 2*bar_width, relaxed_y, bar_width, color='skyblue')
bar3 = ax.bar(index - bar_width, relaxed_z, bar_width, color='skyblue')
bar4 = ax.bar(index + bar_width, excited_x, bar_width, label = 'Excited', color='coral')
bar5 = ax.bar(index + 2*bar_width, excited_y, bar_width, color='coral')
bar6 = ax.bar(index + 3*bar_width, excited_z, bar_width, color='coral')

ax.bar_label(bar1, labels=['x', 'x', 'x'], padding=0)
ax.bar_label(bar2, labels=['y', 'y', 'y'], padding=0)
ax.bar_label(bar3, labels=['z', 'z', 'z'], padding=0)
ax.bar_label(bar4, labels=['x', 'x', 'x'], padding=0)
ax.bar_label(bar5, labels=['y', 'y', 'y'], padding=0)
ax.bar_label(bar6, labels=['z', 'z', 'z'], padding=0)

ax.set_xlabel('User')
ax.set_ylabel('Variance')
ax.set_title('Variance of accelerometer data by user and emotion')
ax.set_xticks(index)
labels = []
for i in range(N_USERS):
    labels.append(f'User {i}')
ax.set_xticklabels(labels)
ax.legend()
plt.show()

# # save plot
# fig.savefig('eeg_classifier_variances.png')


