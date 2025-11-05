import json

def process_jsonl(input_file, output_file):
    """
    处理 JSONL 文件：
    1. 将 'closest_idx' 和 'progress_score' 替换为 'n/a'
    2. 将 'data_source' 中去掉 'robomind_' 前缀
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # 解析 JSON
                data = json.loads(line)

                # 替换 closest_idx 和 progress_score 为 "n/a"
                data['closest_idx'] = 'n/a'
                data['progress_score'] = 'n/a'

                # 去掉 data_source 中的 'robomind_' 前缀
                if 'data_source' in data and data['data_source'].startswith('robomind_'):
                    data['data_source'] = data['data_source'].replace('robomind_', '', 1)

                # 写入处理后的数据
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

            except json.JSONDecodeError as e:
                print(f"警告：第 {line_num} 行解析失败: {e}")
                continue

    print(f"处理完成！输出文件: {output_file}")


if __name__ == '__main__':
    input_path = '/projects/p32958/chengxuan/ProgressLM/data/train/visual_demo/visual_negative_trans_img_replaced.jsonl'
    output_path = '/projects/p32958/chengxuan/ProgressLM/data/train/visual_demo/visual_negative_trans_img_replaced_processed.jsonl'

    process_jsonl(input_path, output_path)
