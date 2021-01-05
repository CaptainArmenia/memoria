import matplotlib.pyplot as plt
import numpy as np

loss = [1.67, 0.89, 0.72, 0.62, 0.54, 0.49, 0.44, 0.39, 0.35, 0.32]
val_loss = [1.29, 0.94, 0.92, 0.87, 0.72, 0.78, 0.94, 0.91, 0.81, 0.61]
accuracy = [0.57, 0.74, 0.79, 0.81, 0.83, 0.85, 0.86, 0.88, 0.89, 0.89]
val_accuracy = [0.69, 0.76, 0.75, 0.76, 0.8,0.81, 0.82, 0.78, 0.79, 0.84]

# Data for plotting
t = np.arange(1, 11, 1)
s = loss

fig, ax = plt.subplots()
s1, = ax.plot(t, accuracy)
s2, = ax.plot(t, val_accuracy)

s1.set_label("accuracy")
s2.set_label("validation accuracy")

ax.legend()

ax.set(xlabel='Epoch', ylabel='Accuracy',
       title='Resultados del entrenamiento', )
ax.grid()

plt.xticks(np.arange(1, 11, 1))

fig.savefig("accuracy.png")
plt.show()


class_names = ["Suspenders", "Accesory Gift Set", "Umbrellas", "Ties", "Salwar and Dupatta", "Rompers", "Sunglasses", "Watches", "Belts", "Socks"]
f1score = [1, 1, 1, 1, 1, 1, 0.99, 0.98, 0.98, 0.97]

# Data for plotting
t = class_names
s = f1score

x = np.arange(len(class_names))

fig, ax = plt.subplots()
s1 = ax.bar(x, f1score, 0.5)

s1.set_label("f1-score")

ax.legend()

ax.set(xlabel='Clase', ylabel='f1-score',
       title='Resultados del entrenamiento', )

plt.xticks(rotation=60)
ax.set_xticks(x)
ax.set_xticklabels(class_names)
plt.ylim(0.95, 1.01)


fig.savefig("resultados_clases.png")
plt.show()






