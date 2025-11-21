import json
import sys

# 슬라이딩 윈도우 설정
WINDOW_SIZE = 80
OVERLAP = 40

def create_sliding_windows(input_file, output_file):
    """
    JSONL 파일을 읽어서 슬라이딩 윈도우를 적용한 새로운 JSONL 파일 생성
    
    Args:
        input_file: 입력 JSONL 파일 경로
        output_file: 출력 JSONL 파일 경로
    """
    
    # 입력 파일에서 모든 JSON 객체 읽기
    packets = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 빈 줄 무시
                try:
                    packets.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSON 파싱 오류: {e}")
                    continue
    
    print(f"총 {len(packets)}개의 패킷을 읽었습니다.")
    
    # 슬라이딩 윈도우 적용
    step = WINDOW_SIZE - OVERLAP  # 이동 간격
    windows = []
    window_id = 1
    
    start_idx = 0
    while start_idx < len(packets):
        end_idx = min(start_idx + WINDOW_SIZE, len(packets))
        
        window_data = {
            "window_id": window_id,
            "sequence_group": packets[start_idx:end_idx]
        }
        
        windows.append(window_data)
        window_id += 1
        start_idx += step
        
        # 마지막 윈도우가 WINDOW_SIZE보다 작으면 종료
        if end_idx == len(packets):
            break
    
    # 결과를 JSONL 형식으로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for window in windows:
            f.write(json.dumps(window, ensure_ascii=False) + '\n')
    
    print(f"\n결과:")
    print(f"- 생성된 윈도우 개수: {len(windows)}")
    print(f"- 윈도우 크기: {WINDOW_SIZE}")
    print(f"- 오버랩: {OVERLAP}")
    print(f"- 이동 간격: {step}")
    print(f"- 출력 파일: {output_file}")
    
    # 각 윈도우의 실제 크기 출력
    print("\n각 윈도우의 패킷 개수:")
    for i, window in enumerate(windows, 1):
        print(f"  Window {i}: {len(window['sequence_group'])}개")


def main():
    if len(sys.argv) < 3:
        print("사용법: python script.py <입력파일.jsonl> <출력파일.jsonl>")
        print("예시: python script.py input.jsonl output.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    create_sliding_windows(input_file, output_file)


if __name__ == "__main__":
    main()