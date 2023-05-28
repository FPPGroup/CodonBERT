import re, os
import pandas as pd
import argparse
import sys

def set_args():
    parser = argparse.ArgumentParser(description='NanoSTR: a STR genotyping tool for forensic STR')       #创建解析器
    ## call
    parser.add_argument('-f', '--fasta', help='the seq file',required = True, type = str)             #添加可选参数，以--开头，如果没有以这个开头，就会报错
    parser.add_argument('-o', '--output_path', help='the save path',required = True, type = str)
    args = parser.parse_args()      #实例化
    return args

def fasta_reader(fasta_path):
    fasta = {}
    with open(fasta_path, 'r') as file:
        for line in file:
            striped = line.strip()             #strip()表示删除掉数据中的空格和换行符
            if striped.startswith('>'):
                seq_name = striped[1:]         #[1:]意思是去掉列表中第一个元素（下标为0），去后面的元素进行操作
                fasta[ seq_name ] = ""
            else:
                fasta[ seq_name ] += striped
    return fasta

def GC_con(fasta_dict):
    gc_dict = {}
    for name, seq in fasta_dict.items():
        length = len(seq)
        G_num = seq.count('G')
        C_num = seq.count('C')
        GC_content = (G_num+C_num)/length
        gc_dict[name] = round(GC_content, 4)
    return gc_dict

def extract_MFE(rnafold_res):
    mfe_dict = {}
    with open(rnafold_res, 'r') as f:
        for ind, line in enumerate(f):
            striped = line.strip()
            if striped.startswith('>'):
                seq_name = striped[1:]
            
            if ind % 7 == 2:
                mfe_score = float( re.findall('[+-]\d+.\d+', line)[0] )         #re.findall(pattern, string, flags=0)：返回string中所有与pattern相匹配的全部字符串
                mfe_dict[seq_name] = mfe_score
                
    return mfe_dict
            
def extract_cai(cai_res):
    cai_dict = {}
    with open(cai_res) as f:
        for line in f:
            seq_name  = re.findall('Sequence: (.*) CAI:', line)[0]             #.*表示任意字符出现零次或多次,()按格式匹配，取（）里面的
            cai_score = float( re.findall('CAI: (\d+.\d+)', line)[0] )
            cai_dict[seq_name] = cai_score
    return cai_dict

def extract_enc(enc_res):
    enc_dict = {}
    with open(enc_res) as f:
        for line in f:
            seq_name  = re.findall('([^\s]+)\s', line)[0]                      #获取非空字符开头，以空字符结束的字符串
            # print(" re.findall('\d+.\d+', line)[0]", re.findall('= (\d+.\d+)', line))
            enc_score = float( re.findall('= (\d+.\d+)', line)[0] )
            enc_dict[seq_name] = enc_score
    return enc_dict

def merge_func(col_names, *dicts):
    col_names  = ['name'] + col_names
    first_dict = dicts[0]
    # print(dicts)
    
    output = []
    for ind, (key, val) in enumerate(first_dict.items()):
        # print([key, val] + [dct[key] for dct in dicts[1:]])
        output.append( [key, val] + [dct[key] for dct in dicts[1:]] )
        
    output = pd.DataFrame(output, columns=col_names)
    return output
##################################################

if __name__ == '__main__':
    args = set_args()
    print(args.fasta)

    # fasta_path  = '/mnt/public2/jiangl/Projects/Project_codon_optim/data/optim_rna/lab_optim/covid_mrna_cds.fasta'
    fasta_path = args.fasta
    temp_file   = "temp.txt"

    ## gc
    fasta_dict = fasta_reader(fasta_path)
    gc_dict = GC_con(fasta_dict)

    ## mfe
    command_line = "/mnt/public2/jiangl/miniconda3/envs/RNA_index_cal/bin/RNAfold -p --MEA < %s > %s" % (fasta_path, temp_file)
    os.system(command_line)
    mfe_dict = extract_MFE(temp_file) ## mfe scores are inconsistent with free energy; TODO: need to figure out
    os.remove(temp_file)
    if any(name.endswith(('.ps')) for name in os.listdir(sys.path[0])):
        r = os.system('/usr/bin/find ./ -type f -name "*.ps" | xargs /usr/bin/rm')
        # os.remove('*.ps')

    ## cai
    command_line = "/mnt/public2/jiangl/miniconda3/envs/RNA_index_cal/bin/_cai -seqall %s -cfile Ehuman.cut -outfile %s" % (fasta_path, temp_file) # Ehuman.cut is default
    os.system(command_line)
    cai_dict = extract_cai(temp_file)
    os.remove(temp_file)

    ## enc
    command_line = "/mnt/public2/jiangl/miniconda3/envs/RNA_index_cal/bin/_chips -seqall %s -outfile %s -nosum" % (fasta_path, temp_file)
    os.system(command_line)
    enc_dict = extract_enc(temp_file)
    os.remove(temp_file)

    metrics_result = merge_func(['GC', 'MFE', 'CAI', 'ENC'], gc_dict, mfe_dict, cai_dict, enc_dict)
    # print(metrics_result)
    seq_name = list(gc_dict.keys())[0].split('_')[0]
    save_path = args.output_path+seq_name+"_metrics_result.csv"
    metrics_result.to_csv(save_path,index=False,header=True)