import subprocess
import os
import random
import time
import ltspice
import numpy as np
import openpyxl
import sys



ltspice_path = r"/Applications/LTspice.app/Contents/MacOS/LTspice"



def run_sim_extract_gain(ltspice_path, circuit_path):
    try:
        # LTspice 프로세스를 백그라운드에서 실행
        ltspice_process = subprocess.Popen([ltspice_path, "-Run", circuit_path])
        # 결과 파일이 생성될 때까지 대기
        output_path = os.path.splitext(circuit_path)[0] + ".raw"
        elapsed_time = 0
        timeout = 2

        file_size = -1  # 파일 크기 초기화

        while elapsed_time < timeout:
            if os.path.isfile(output_path):
                new_size = os.path.getsize(output_path)
                if new_size != file_size:
                    file_size = new_size
                    elapsed_time = 0  # 파일 크기가 변하면 타이머 초기화
                else:
                    # 파일 크기가 변하지 않으면 기다림
                    print("Waiting for output file...gain")
                    time.sleep(0.3)
                    elapsed_time += 0.3
            else:
                print("Output file not found. Waiting...")
                time.sleep(0.3)
                elapsed_time += 0.3

        if not os.path.isfile(output_path):
            print(f"Output file not found after {timeout} seconds.")
            ltspice_process.terminate()
            return 0

        l = ltspice.Ltspice(output_path)
        l.parse()

        voutPos = l.get_data("V(vout+)")
        voutNeg = l.get_data("V(vout-)")
        vin = l.get_data("V(n010)")
        
        Ivdd = l.get_data("I(Vdd)")
        Vn001 = l.get_data("V(n001)")

        # LTspice 프로세스 종료 확인
        while ltspice_process.poll() is None:
            # print("Waiting for LTspice process to terminate...")
            ltspice_process.terminate()  # 시그널을 보내 LTspice를 종료시킴
            time.sleep(1)

        #gain 계산
        voutPos_values = voutPos.data
        voutNeg_values = voutNeg.data
        vin_values = vin.data

        vout_diff = np.max(voutPos_values) - np.min(voutNeg_values)
        vin_peak = np.max(vin_values)

        voltage_gain = round(np.abs(vout_diff) / np.abs(vin_peak), 3)

        #Power Consumption 계산
        Ivdd_values = np.array(Ivdd)
        Vn001_values = np.array(Vn001)

        product = np.abs(Ivdd_values * Vn001_values)
        power_consumption = round((np.mean(product)), 3)

        return voltage_gain, power_consumption

    except ltspice.ltspice.FileSizeNotMatchException:
        print("File size not matching, skipping this simulation. gain")
        ltspice_process.terminate()
        time.sleep(1)
        # sys.exit(1)  # 전체 프로그램을 종료하는 코드 추가

    return 0, 10000  # Return 0 values to indicate a skipped simulation


def run_sim_extract_noise_figure(ltspice_path, circuit_path, target_frequency, timeout=3):
    # LTspice 프로세스를 백그라운드에서 실행
    ltspice_process = subprocess.Popen([ltspice_path, "-Run", circuit_path])
    # 결과 파일이 생성될 때까지 대기
    output_path = os.path.splitext(circuit_path)[0] + ".raw"
    elapsed_time = 0
    timeout = 2

    file_size = -1  # 파일 크기 초기화

    while elapsed_time < timeout:
        if os.path.isfile(output_path):
            new_size = os.path.getsize(output_path)
            if new_size != file_size:
                file_size = new_size
                elapsed_time = 0  # 파일 크기가 변하면 타이머 초기화
            else:
                # 파일 크기가 변하지 않으면 기다림
                print("Waiting for output file...gain")
                time.sleep(0.3)
                elapsed_time += 0.3
        else:
            print("Output file not found. Waiting...")
            time.sleep(0.3)
            elapsed_time += 0.3

    if not os.path.isfile(output_path):
        print(f"Output file not found after {timeout} seconds.")
        ltspice_process.terminate()
        return 1


    # print(f"Output file exists at {output_path}")

    l = ltspice.Ltspice(output_path)
    l.parse()

    frequency = l.get_frequency()

    vonoise = l.get_data("V(onoise)")
    vin = l.get_data("V(vin)")

    while ltspice_process.poll() is None:
        # print("Waiting for LTspice process to terminate...")
        ltspice_process.terminate()
        time.sleep(1)

    if vonoise is not None and vin is not None:
        # Find the index closest to the target frequency (2.4 GHz)
        idx = np.argmin(np.abs(frequency - target_frequency))

        # Extract V(onoise) and V(vin) at the target frequency
        vonoise_at_target_freq = vonoise.data[idx]
        vin_at_target_freq = vin.data[idx]

        # Calculate the noise figure at the target frequency
        noise_figure_at_target_freq = round(20 * np.log10(np.abs(vonoise_at_target_freq / vin_at_target_freq)), 3)

        # if noise_figure_at_target_freq > 3:
        #     noise_figure_at_target_freq = 100000
        return noise_figure_at_target_freq

    return 1  # Handle the case where simulation results are missing or skipped

def modify_circuit_file(circuit_path, individual):
    # 변수 초기화
    r1, r2, r3, r4, rm, c1, c2, c3, vb, VLO_off, VLO_amp, Vin = individual

    # 텍스트 파일 열기
    with open(circuit_path, 'r') as f:
        lines = f.readlines()

    # 행을 순회하면서 원하는 패턴 찾기
    for i, line in enumerate(lines):
        if "SYMATTR InstName R1" in line:
            lines[i+1] = f'SYMATTR Value {r1}\n'
        elif "SYMATTR InstName R2" in line:
            lines[i+1] = f'SYMATTR Value {r2}\n'
        elif "SYMATTR InstName R3" in line:
            lines[i+1] = f'SYMATTR Value {r3}\n'
        elif "SYMATTR InstName R4" in line:
            lines[i+1] = f'SYMATTR Value {r4}\n'
        elif "SYMATTR InstName Rm" in line:
            lines[i+1] = f'SYMATTR Value {rm}\n'
        elif "SYMATTR InstName C1" in line:
            lines[i+1] = f'SYMATTR Value {c1}\n'
        elif "SYMATTR InstName C2" in line:
            lines[i+1] = f'SYMATTR Value {c2}\n'
        elif "SYMATTR InstName C3" in line:
            lines[i+1] = f'SYMATTR Value {c3}\n'
        elif "SYMATTR InstName Vb" in line:
            lines[i+1] = f'SYMATTR Value {vb}\n'
        elif "SYMATTR InstName VLO+" in line:
            lines[i+1] = f'SYMATTR Value SINE({VLO_off} {VLO_amp} 2.3G 0 0 180 0)\n'
        elif "SYMATTR InstName VLO-" in line:
            lines[i+1] = f'SYMATTR Value SINE({VLO_off} {VLO_amp} 2.3G 0 0 180 0)\n'
        elif "SYMATTR InstName Vin" in line:
            lines[i-2] = f'SYMATTR Value SINE(0 {Vin} 2.4G)\n'

    updated_circuit = ''.join(lines)
    updated_circuit_path = os.path.splitext(circuit_path)[0] + "_updated.asc"
    with open(updated_circuit_path, "w") as f:
        f.write(updated_circuit)
    # 변경된 변수 반환
    return updated_circuit_path



def remove_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
    time.sleep(0.5)

def simulate_and_evaluate(individual, ltspice_path, gain_circuit_path, noise_figure_circuit_path):

    # Gain circuit 파일 수정
    gain_updated_circuit_path = modify_circuit_file(gain_circuit_path, individual)
    # Power consumption circuit 파일 수정

    # Noise figure circuit 파일 수정
    noise_figure_updated_circuit_path = modify_circuit_file(noise_figure_circuit_path, individual)

    # LTspice 시뮬레이션 실행 및 결과 분석
    gain, powerConsumption = run_sim_extract_gain(ltspice_path, gain_updated_circuit_path)
    noiseFigure = run_sim_extract_noise_figure(ltspice_path, noise_figure_updated_circuit_path, 2.4e9)

    # 업데이트된 회로 파일 삭제
    os.remove(gain_updated_circuit_path)
    os.remove(noise_figure_updated_circuit_path)

    # 파일 경로 리스트 정의
    file_paths_to_remove = [
        '/Users/mingikwon/Documents/LTspice/grad/grad_gain_updated.raw',
        '/Users/mingikwon/Documents/LTspice/grad/grad_gain_updated.op.raw',
        '/Users/mingikwon/Documents/LTspice/grad/grad_gain_updated.net',
        '/Users/mingikwon/Documents/LTspice/grad/grad_gain_updated.log',
        '/Users/mingikwon/Documents/LTspice/grad/grad_noise_updated.raw',
        '/Users/mingikwon/Documents/LTspice/grad/grad_noise_updated.op.raw',
        '/Users/mingikwon/Documents/LTspice/grad/grad_noise_updated.net',
        '/Users/mingikwon/Documents/LTspice/grad/grad_noise_updated.log']
    
    # 파일 삭제 함수 호출
    remove_files(file_paths_to_remove)

    return gain, powerConsumption, noiseFigure



# Genetic Algorithm 초기화
def create_individual():
    r1 = random.randint(3000, 6000)
    r2 = random.randint(1500, 4000)
    r3 = random.randint(1, 1000)
    r4 = random.randint(1000, 100000)
    rm = random.randint(100, 1000)
    c1 = str(round(random.uniform(1.5, 3), 2)) + 'p'
    c2 = str(round(random.uniform(0.1, 5), 2)) + 'p'
    c3 = str(round(random.uniform(0.1, 1), 2)) + 'p'
    vb = round(random.uniform(0.1, 0.4), 3)
    VLO_off = round(random.uniform(0, 0.5), 3)
    VLO_amp = round(random.uniform(0, 1), 3)
    Vin = round(random.uniform(0, 0.5), 3)
    
    return [r1, r2, r3, r4, rm, c1, c2, c3, vb, VLO_off, VLO_amp, Vin]

def mutate_individual(individual):
    mutated = individual.copy()
    index = random.randint(0, 11)  # 12개의 요소 중 하나를 선택

    if index == 0:
        mutated[index] = random.randint(3000, 6000)  # r1 값 변경
    elif index == 1:
        mutated[index] = random.randint(1500, 4000)  # r2 값 변경
    elif index == 2:
        mutated[index] = random.randint(1, 1000)    # r3 값 변경
    elif index == 3:
        mutated[index] = random.randint(1000, 100000)  # r4 값 변경
    elif index == 4:
        mutated[index] = random.randint(100, 1000)   # rm 값 변경
    elif index == 5:
        mutated[index] = str(round(random.uniform(1.5, 3), 2)) + 'p'  # c1 값 변경
    elif index == 6:
        mutated[index] = str(round(random.uniform(0.1, 5), 2)) + 'p'  # c2 값 변경
    elif index == 7:
        mutated[index] = str(round(random.uniform(0.1, 1), 2)) + 'p'  # c3 값 변경
    elif index == 8:
        mutated[index] = round(random.uniform(0.1, 0.4), 3)  # vb 값 변경
    elif index == 9:
        mutated[index] = round(random.uniform(0, 0.5), 3)    # VLO_off 값 변경
    elif index == 10:
        mutated[index] = round(random.uniform(0, 1), 3)    # VLO_amp 값 변경
    else:
        mutated[index] = round(random.uniform(0, 0.5), 3)  # Vin 값 변경

    return mutated

def mutate_individual_12(individual, num):
    mutated = individual.copy()
    index = num

    if index == 0:
        mutated[index] = random.randint(3000, 6000)  # r1 값 변경
    elif index == 1:
        mutated[index] = random.randint(1500, 4000)  # r2 값 변경
    elif index == 2:
        mutated[index] = random.randint(1, 1000)    # r3 값 변경
    elif index == 3:
        mutated[index] = random.randint(1000, 100000)  # r4 값 변경
    elif index == 4:
        mutated[index] = random.randint(100, 1000)   # rm 값 변경
    elif index == 5:
        mutated[index] = str(round(random.uniform(1.5, 3), 2)) + 'p'  # c1 값 변경
    elif index == 6:
        mutated[index] = str(round(random.uniform(0.1, 5), 2)) + 'p'  # c2 값 변경
    elif index == 7:
        mutated[index] = str(round(random.uniform(0.1, 1), 2)) + 'p'  # c3 값 변경
    elif index == 8:
        mutated[index] = round(random.uniform(0.1, 0.4), 3)  # vb 값 변경
    elif index == 9:
        mutated[index] = round(random.uniform(0, 0.5), 3)    # VLO_off 값 변경
    elif index == 10:
        mutated[index] = round(random.uniform(0, 1), 3)    # VLO_amp 값 변경
    else:
        mutated[index] = round(random.uniform(0, 0.5), 3)  # Vin 값 변경

    return mutated

excel_file_path = 'results.xlsx'
wb = openpyxl.Workbook()
ws = wb.active
ws.append(['Generation', 'Simulations Run', 'FOM', 'Individual'])

gain_file_path = r"/Users/mingikwon/Documents/LTspice/grad/grad_gain.asc"
powerConsumption_file_path = r"/Users/mingikwon/Documents/LTspice/grad/grad_power.asc"
noiseFigure_file_path = r"/Users/mingikwon/Documents/LTspice/grad/grad_noise.asc"

# 최적 결과 찾기
best_FOM = 0
best_individual = None
simulations_run = 0
target_FOM = 10000
max_simulations = 200

n_gen = 5 # 세대 수
pop_size = 27  # 개체 집단 크기

population = [create_individual() for _ in range(pop_size)]

# 1. Best 만 갖고 계속 변화

for generation in range(n_gen):
    # 개체 평가와 선택
    print('generation:', generation)
    evaluated_population = []
    for ind in population:
        gain, powerConsumption, noiseFigure = simulate_and_evaluate(ind, ltspice_path, gain_file_path, noiseFigure_file_path)
        simulations_run += 1
        print(simulations_run)
        FOM = round(gain / (noiseFigure * powerConsumption), 3)

        if gain != 0 and powerConsumption != 10000:
            evaluated_population.append((ind, gain, powerConsumption, noiseFigure, FOM))  # FOM 추가
            individual_str = ', '.join(map(str, ind))
            ws.append([generation, simulations_run, FOM, individual_str])
            wb.save(excel_file_path)

        print('FOM:', FOM, 'Gain:', gain, 'power:', powerConsumption, 'NF:', noiseFigure)
        if FOM > best_FOM:
            best_FOM = FOM
            best_individual = ind
            print('best FOM and individual', FOM, ind)

    # 개체 평가 기준에 따라 정렬 (FOM이 높은 순서로 정렬)
    evaluated_population.sort(key=lambda x: -x[4])  # FOM을 기준으로 정렬

    print("evaluated_population 개수:", len(evaluated_population))

    # 최상위 개체 선택
    new_pop_size = len(evaluated_population)
    print('new pop size', new_pop_size)
    top_individuals = evaluated_population[0]

    # 변이와 다음 세대 생성
    new_population = []

    ind, _, _, _, _ = top_individuals
    for i in range(12):  # FOM 값을 추가하였으므로 인덱스 수정
        mutated_ind = mutate_individual_12(ind, i)
        new_population.append(mutated_ind)

    # 종료 조건 검사 (예: 최대 세대 수나 목표 FOM 달성 여부)
    if best_FOM >= target_FOM or simulations_run >= max_simulations:
        break

    if new_pop_size == 1:
        break
    #  다음 세대로 업데이트
    population = new_population


    
# # 2. 일반 코드

# for generation in range(n_gen):
#     # 개체 평가와 선택
#     print('generation:', generation)
#     evaluated_population = []
#     for ind in population:
#         gain, powerConsumption, noiseFigure = simulate_and_evaluate(ind, ltspice_path, gain_file_path, noiseFigure_file_path)
#         simulations_run += 1
#         print(simulations_run)
#         FOM = round(gain / (noiseFigure * powerConsumption), 3)

#         if gain != 0 and powerConsumption != 10000:
#             evaluated_population.append((ind, gain, powerConsumption, noiseFigure, FOM))  # FOM 추가
#             individual_str = ', '.join(map(str, ind))
#             ws.append([generation, simulations_run, FOM, individual_str])
#             wb.save(excel_file_path)

#         print('FOM:', FOM, 'Gain:', gain, 'power:', powerConsumption, 'NF:', noiseFigure)
#         if FOM > best_FOM:
#             best_FOM = FOM
#             best_individual = ind
#             print('best FOM and individual', FOM, ind)

#     # 개체 평가 기준에 따라 정렬 (FOM이 높은 순서로 정렬)
#     evaluated_population.sort(key=lambda x: -x[4])  # FOM을 기준으로 정렬

#     print("evaluated_population 개수:", len(evaluated_population))

#     # 상위 개체 선택 (33%)
#     new_pop_size = len(evaluated_population)
#     print('new pop size', new_pop_size)
#     top_individuals = evaluated_population[:new_pop_size // 3]

#     # 변이와 다음 세대 생성
#     new_population = []
#     for ind, _, _, _, _ in top_individuals:  # FOM 값을 추가하였으므로 인덱스 수정
#         mutated_ind = mutate_individual(ind)
#         new_population.append(mutated_ind)

#     # 종료 조건 검사 (예: 최대 세대 수나 목표 FOM 달성 여부)
#     if best_FOM >= target_FOM or simulations_run >= max_simulations:
#         break

#     if new_pop_size == 1:
#         break

#      다음 세대로 업데이트
#     population = new_population

## 3. mutate 할 때 FOM 무조건 증가하는 코드

# for generation in range(n_gen):
#     # 개체 평가와 선택
#     print('generation:', generation)
#     evaluated_population = []
#     for ind in population:
#         gain, powerConsumption, noiseFigure = simulate_and_evaluate(ind, ltspice_path, gain_file_path, noiseFigure_file_path)
#         simulations_run += 1
#         print(simulations_run)
#         FOM = round(gain / (noiseFigure * powerConsumption), 3)

#         if gain != 0 and powerConsumption != 10000:
#             evaluated_population.append((ind, gain, powerConsumption, noiseFigure, FOM))
#             individual_str = ', '.join(map(str, ind))
#             ws.append([generation, simulations_run, FOM, individual_str])
#             wb.save(excel_file_path)

#         print('FOM:', FOM, 'Gain:', gain, 'power:', powerConsumption, 'NF:', noiseFigure)
#         if FOM > best_FOM:
#             best_FOM = FOM
#             best_individual = ind
#             print('best FOM and individual', FOM, ind)

#     # 개체 평가 기준에 따라 정렬 (FOM이 높은 순서로 정렬)
#     evaluated_population.sort(key=lambda x: -x[4])  # FOM을 기준으로 정렬

#     print("evaluated_population 개수:", len(evaluated_population))

#     # 상위 개체 선택 (33%)
#     new_pop_size = len(evaluated_population)
#     print('new pop size', new_pop_size)
#     top_individuals = evaluated_population[:new_pop_size // 3]

#     # 변이와 다음 세대 생성
#     new_population = []
#     for ind, _, _, _, _ in top_individuals:
#         # FOM 개선 여부 확인을 위한 반복 루프
#         while True:
#             mutated_ind = mutate_individual(ind)
#             mutated_gain, mutated_powerConsumption, mutated_noiseFigure = simulate_and_evaluate(mutated_ind, ltspice_path, gain_file_path, noiseFigure_file_path)
#             simulations_run += 1
#             mutated_FOM = round(mutated_gain / (mutated_noiseFigure * mutated_powerConsumption), 3)
#             if mutated_gain != 0 and mutated_powerConsumption != 10000 and mutated_FOM > FOM:
#                 # FOM 개선됨, 새로운 개체로 업데이트
#                 ind = mutated_ind
#                 FOM = mutated_FOM
#                 break

#         new_population.append(ind)

#     # 종료 조건 검사 (예: 최대 세대 수나 목표 FOM 달성 여부)
#     if best_FOM >= target_FOM or simulations_run >= max_simulations:
#         break

#     if new_pop_size == 1:
#         break

#     # 다음 세대로 업데이트
#     population = new_population




# 결과 출력
print("최적의 FOM:", best_FOM)
print("요소 값들:", best_individual)
print("시뮬레이션 실행 횟수:", simulations_run)
wb.save(excel_file_path)

