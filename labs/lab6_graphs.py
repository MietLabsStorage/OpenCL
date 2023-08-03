import numpy as np
import matplotlib.pyplot as plt

file = open("lab6.txt", "r")
lines = []
while True:
  line = file.readline().split()
  if not line:
      break
  lines.append(line)

err_by_row_anf_column_n2 = []
time_by_row_anf_column_n2 = []
err_by_row_n2 = []
time_by_row_n2 = []
err_by_column_n2 = []
time_by_column_n2 = []

err_by_row_anf_column_n4 = []
time_by_row_anf_column_n4 = []
err_by_row_n4 = []
time_by_row_n4 = []
err_by_column_n4 = []
time_by_column_n4 = []

err_by_row_anf_column_n8 = []
time_by_row_anf_column_n8 = []
err_by_row_n8 = []
time_by_row_n8 = []
err_by_column_n8 = []
time_by_column_n8 = []

for line in lines:
  if 'by_row_and_by_col' in line[4]:
    if line[1] == '2':
      err_by_row_anf_column_n2.append(line[2])
      time_by_row_anf_column_n2.append(line[3])
    if line[1] == '4':
      err_by_row_anf_column_n4.append(line[2])
      time_by_row_anf_column_n4.append(line[3])
    if line[1] == '8':
      err_by_row_anf_column_n8.append(line[2])
      time_by_row_anf_column_n8.append(line[3])

for line in lines:
  if 'by_row' in line[4] and 'by_col' not in line[4]:
    if line[1] == '2':
      err_by_row_n2.append(line[2])
      time_by_row_n2.append(line[3])
    if line[1] == '4':
      err_by_row_n4.append(line[2])
      time_by_row_n4.append(line[3])
    if line[1] == '8':
      err_by_row_n8.append(line[2])
      time_by_row_n8.append(line[3])

for line in lines:
  if 'by_row' not in line[4] and 'by_col' in line[4]:
    if line[1] == '2':
      err_by_column_n2.append(line[2])
      time_by_column_n2.append(line[3])
    if line[1] == '4':
      err_by_column_n4.append(line[2])
      time_by_column_n4.append(line[3])
    if line[1] == '8':
      err_by_column_n8.append(line[2])
      time_by_column_n8.append(line[3])

x = [8, 16, 32, 64, 128, 256, 512]
fig1, ax1 = plt.subplots()

ax1.plot(x, err_by_row_anf_column_n2, label='by_row_anf_column_n2')
ax1.plot(x, err_by_row_anf_column_n4, label='by_row_anf_column_n4')
ax1.plot(x, err_by_row_anf_column_n8, label='by_row_anf_column_n8')
plt.legend()

fig2, ax2 = plt.subplots()

ax2.plot(x, err_by_row_n2, label='by_row_n2')
ax2.plot(x, err_by_row_n4, label='by_row_n4')
ax2.plot(x, err_by_row_n8, label='by_row_n8')
plt.legend()


fig3, ax3 = plt.subplots()


ax3.plot(x, err_by_column_n2, label='by_column_n2')
ax3.plot(x, err_by_column_n4, label='by_column_n4')
ax3.plot(x, err_by_column_n8, label='by_column_n8')
plt.legend()






ax1.title.set_text('# err')
ax2.title.set_text('# err')
ax3.title.set_text('# err')
fig1.set_size_inches(15, 6)
fig2.set_size_inches(15, 6)
fig3.set_size_inches(15, 6)

fig4, ax4 = plt.subplots()
ax4.plot(x, time_by_row_anf_column_n2, label='by_row_anf_column_n2')
ax4.plot(x, time_by_row_anf_column_n4, label='by_row_anf_column_n4')
ax4.plot(x, time_by_row_anf_column_n8, label='by_row_anf_column_n8')
plt.legend()


fig5, ax5 = plt.subplots()
ax5.plot(x, time_by_row_n2, label='by_row_n2')
ax5.plot(x, time_by_row_n4, label='by_row_n4')
ax5.plot(x, time_by_row_n8, label='by_row_n8')
plt.legend()


fig6, ax6 = plt.subplots()
ax6.plot(x, time_by_column_n2, label='by_column_n2')
ax6.plot(x, time_by_column_n4, label='by_column_n4')
ax6.plot(x, time_by_column_n8, label='by_column_n8')
plt.legend()
ax4.title.set_text('# time')
ax5.title.set_text('# time')
ax5.title.set_text('# time')

fig4.set_size_inches(15, 6)
fig5.set_size_inches(15, 6)
fig6.set_size_inches(15, 6)
plt.legend()
plt.show()
