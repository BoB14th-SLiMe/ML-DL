import json
import sys  # 🌟 명령줄 인자를 받기 위해 sys 모듈 임포트

def merge_packets_by_modbus_diff(logs):
    """
    @timestamp와 sq 기준으로 로그를 그룹화합니다.
    그룹 내에서 modbus.* 키의 값이 다를 경우 리스트로 병합합니다.
    """
    
    grouped_packets = {}
    for log in logs:
        ts = log.get('@timestamp', '')
        sq = log.get('sq', '')
        group_key = f"{ts}_{sq}"
        
        if group_key not in grouped_packets:
            grouped_packets[group_key] = []
        
        grouped_packets[group_key].append(log)

    final_logs = []
    
    for group_key, packet_list in grouped_packets.items():
        
        if len(packet_list) == 1:
            final_logs.append(packet_list[0])
            continue

        base_packet = packet_list[0].copy()
        all_keys_in_group = set()
        for pkt in packet_list:
            all_keys_in_group.update(pkt.keys())

        for key in sorted(list(all_keys_in_group)):
            
            if key.startswith('modbus.'):
                values_list = [pkt.get(key) for pkt in packet_list]
                first_val = values_list[0]
                all_same = True
                for val in values_list[1:]:
                    if val != first_val:
                        all_same = False
                        break
                
                if all_same:
                    base_packet[key] = first_val
                else:
                    base_packet[key] = values_list
            else:
                pass 
        
        final_logs.append(base_packet)
        
    return final_logs

if len(sys.argv) != 3:
    print(f"(!) 사용법: python3 {sys.argv[0]} <입력_jsonl_파일> <출력_jsonl_파일>")
    print(f"    예시: python3 {sys.argv[0]} logs.jsonl output1.jsonl")
    sys.exit(1) # 오류로 종료

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

raw_logs = []

try:
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    raw_logs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSON 파싱 오류 (입력): {line}")

except FileNotFoundError:
    print(f"!!! 오류: '{input_file_path}' 파일을 찾을 수 없습니다.")
    sys.exit(1)
except Exception as e:
    print(f"!!! 오류: '{input_file_path}' 파일 읽기 중 오류 발생: {e}")
    sys.exit(1)

if raw_logs:
    print(f"======= 총 {len(raw_logs)} 줄의 데이터를 '{input_file_path}'에서 읽었습니다. =======")
    merged_logs = merge_packets_by_modbus_diff(raw_logs)

    # --- 3. 지정된 파일로 저장 ---
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            for log_entry in merged_logs:
                json_line = json.dumps(log_entry, ensure_ascii=False)
                f_out.write(json_line + '\n')
        
        print(f"======= {len(merged_logs)}개로 병합 완료. '{output_file_path}' 파일로 저장되었습니다. =======")

    except IOError as e:
        print(f"!!! 오류: '{output_file_path}' 파일 쓰기 중 오류 발생: {e}")
    except Exception as e:
        print(f"!!! 알 수 없는 오류 발생: {e}")

else:
    print(f"!!! 경고: '{input_file_path}' 파일에서 데이터를 읽지 못했습니다. 파일이 비어있는지 확인하세요.")