import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt

# --- Подключение ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# --- Загрузка модели ---
robot = p.loadURDF(r"c:\\Users\\devya\\Desktop\\nir22\\triple_pendulum.urdf", useFixedBase=True)

# --- Револьвентные суставы ---
revolute_joints = [1, 3, 5]
print("Revolute joints:", revolute_joints)

# --- Отключаем встроенные моторы ---
for j in revolute_joints:
    p.setJointMotorControl2(
        bodyIndex=robot,
        jointIndex=j,
        controlMode=p.VELOCITY_CONTROL,
        force=0
    )

# --- Настройки симуляции ---
p.setTimeStep(1.0 / 240.0)
steps = 5000
dt = 1.0 / 240.0

# --- Задаем траектории ---
def desired_trajectory(t):
    qd = [0.3*np.sin(0.5*t),
          0.4*np.sin(0.5*t + np.pi/4),
          0.2*np.sin(0.5*t + np.pi/2)]
    qd_dot = [0.3*0.5*np.cos(0.5*t),
              0.4*0.5*np.cos(0.5*t + np.pi/4),
              0.2*0.5*np.cos(0.5*t + np.pi/2)]
    qd_ddot = [-0.3*(0.5**2)*np.sin(0.5*t),
               -0.4*(0.5**2)*np.sin(0.5*t + np.pi/4),
               -0.2*(0.5**2)*np.sin(0.5*t + np.pi/2)]
    return qd, qd_dot, qd_ddot

# --- ПИД-параметры ---
Kp, Kd = 50, 5
damping_coeff = 1.0   # из URDF

# --- Логирование ---
q_log, qd_log = [], []
q_my_log = []

# --- Начальные условия для своей модели ---
q_my = np.array([0.0, 0.0, 0.0])
qd_my = np.array([0.0, 0.0, 0.0])

# --- Основной цикл ---
for step in range(steps):
    t = step * dt

    # --- PyBullet ---
    q = [p.getJointState(robot, j)[0] for j in revolute_joints]
    qdot = [p.getJointState(robot, j)[1] for j in revolute_joints]

    qd, qd_dot, qd_ddot = desired_trajectory(t)

    # Управляющее ускорение (PD + feedforward)
    qdd = [qddi - Kp*(qi - qdi) - Kd*(dqi - dqdi)
           for qi, dqi, qdi, dqdi, qddi in zip(q, qdot, qd, qd_dot, qd_ddot)]

    # --- PyBullet inverse dynamics ---
    tau = p.calculateInverseDynamics(robot, q, qdot, qdd)

    # Применяем моменты
    for j, t_j in zip(revolute_joints, tau):
        p.setJointMotorControl2(
            bodyUniqueId=robot,
            jointIndex=j,
            controlMode=p.TORQUE_CONTROL,
            force=t_j
        )

    # --- Собственная модель с semi-implicit Euler + демпфирование ---
    # Управляющее ускорение без учёта демпфирования
    qdd_my = np.array([qddi - Kp*(qi - qdi) - Kd*(dqi - dqdi)
                       for qi, dqi, qdi, dqdi, qddi in zip(q_my, qd_my, qd, qd_dot, qd_ddot)])

    # Добавляем эффект вязкого демпфирования (a_damp = -b * v)
    qdd_my -= damping_coeff * qd_my

    # Semi-implicit Euler:
    qd_my = qd_my + qdd_my * dt
    q_my = q_my + qd_my * dt

    # Логирование
    q_log.append(q)
    qd_log.append(qd)
    q_my_log.append(q_my.copy())

    p.stepSimulation()

print("Симуляция завершена")

# --- Перевод в numpy ---
q_log = np.array(q_log)
qd_log = np.array(qd_log)
q_my_log = np.array(q_my_log)
time_axis = np.linspace(0, steps*dt, steps)

# --- Метрики ---
error = q_log - q_my_log
L2 = np.sqrt(np.mean(np.sum(error**2, axis=1)))
Linf = np.max(np.abs(error))
print(f"Метрики сравнения PyBullet vs своя модель :")
print(f"L2-норма ошибки = {L2:.4f} рад")
print(f"L∞-норма ошибки = {Linf:.4f} рад")

# --- Построение графиков ---
plt.figure(figsize=(9,6))
plt.plot(time_axis, q_log[:,0], label="PyBullet q1")

plt.plot(time_axis, q_log[:,1], label="PyBullet q2")

plt.plot(time_axis, q_log[:,2], label="PyBullet q3")
plt.plot(time_axis, qd_log[:,0], '--', label="Target q1")
plt.plot(time_axis, qd_log[:,1], '--', label="Target q2")
plt.plot(time_axis, qd_log[:,2], '--', label="Target q3")

plt.xlabel("Время, c")
plt.ylabel("Углы (рад)")
plt.legend()
plt.title("PyBullet")
plt.grid()
plt.show()

plt.figure(figsize=(9,6))

plt.plot(time_axis, q_my_log[:,0], '--', label="MyModel q1")

plt.plot(time_axis, q_my_log[:,1], '--', label="MyModel q2")

plt.plot(time_axis, q_my_log[:,2], '--', label="MyModel q3")
plt.plot(time_axis, qd_log[:,0], '--', label="Target q1")
plt.plot(time_axis, qd_log[:,1], '--', label="Target q2")
plt.plot(time_axis, qd_log[:,2], '--', label="Target q3")
plt.xlabel("Время, c")
plt.ylabel("Углы (рад)")
plt.legend()
plt.title("Собственная модель")
plt.grid()
plt.show()


mask = time_axis >= 2.0
error_pb = q_log[mask] - qd_log[mask]
L2_pb = np.sqrt(np.mean(np.sum(error_pb**2, axis=1)))
Linf_pb = np.max(np.abs(error_pb))
print(f"Нормы PyBullet vs Target (t >= 2c):")
print(f"L2-норма ошибки = {L2_pb:.4f} рад")
print(f"L∞-норма ошибки = {Linf_pb:.4f} рад")   


error_my = q_my_log[mask] - qd_log[mask]
L2_my = np.sqrt(np.mean(np.sum(error_my**2, axis=1)))
Linf_my = np.max(np.abs(error_my))
print(f"нормы MyModel vs Target (t >= 2c):")
print(f"L2-норма ошибки = {L2_my:.4f} рад")
print(f"L∞-норма ошибки = {Linf_my:.4f} рад")

error = q_log[mask] - q_my_log[mask]
L2 = np.sqrt(np.mean(np.sum(error**2, axis=1)))
Linf = np.max(np.abs(error))
print(f"Нормы сравнения PyBullet vs своя модель (t >= 2c):")
print(f"L2-норма ошибки = {L2:.4f} рад")
print(f"L∞-норма ошибки = {Linf:.4f} рад")
