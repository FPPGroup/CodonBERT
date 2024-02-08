import re, os
import pandas as pd
import argparse
import sys

def set_args():
    parser = argparse.ArgumentParser(description='NanoSTR: a STR genotyping tool for forensic STR')      
    parser.add_argument('-e', '--env_path', help='environment absolute path',required = True, type = str)
    parser.add_argument('-f', '--fasta', help='mRNA fasta',required = True, type = str)            
    parser.add_argument('-o', '--output_path', help='metrics result path',required = True, type = str)
    args = parser.parse_args()     
    return args

def fasta_reader(fasta_path):
    fasta = {}
    with open(fasta_path, 'r') as file:
        for line in file:
            striped = line.strip()             
            if striped.startswith('>'):
                seq_name = striped[1:]        
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
                mfe_score = float( re.findall('[+-]\d+.\d+', line)[0] )        
                mfe_dict[seq_name] = mfe_score
                
    return mfe_dict
            
def extract_cai(cai_res):
    cai_dict = {}
    with open(cai_res) as f:
        for line in f:
            seq_name  = re.findall('Sequence: (.*) CAI:', line)[0]           
            cai_score = float( re.findall('CAI: (\d+.\d+)', line)[0] )
            cai_dict[seq_name] = cai_score
    return cai_dict

def extract_enc(enc_res):
    enc_dict = {}
    with open(enc_res) as f:
        for line in f:
            seq_name  = re.findall('([^\s]+)\s', line)[0]                    
            enc_score = float( re.findall('= (\d+.\d+)', line)[0] )
            enc_dict[seq_name] = enc_score
    return enc_dict

def merge_func(col_names, *dicts):
    col_names  = ['name'] + col_names
    first_dict = dicts[0]
    output = []
    for ind, (key, val) in enumerate(first_dict.items()):
        output.append( [key, val] + [dct[key] for dct in dicts[1:]] )
    output = pd.DataFrame(output, columns=col_names)
    return output


if __name__ == '__main__':
    args = set_args()
    fasta_path = args.fasta
    env_path = args.env_path
    temp_file   = "temp.txt"
    fasta_dict = fasta_reader(fasta_path)
    gc_dict = GC_con(fasta_dict)
    command_line = env_path+"/bin/RNAfold -p --MEA < %s > %s" % (fasta_path, temp_file)
    os.system(command_line)
    mfe_dict = extract_MFE(temp_file) 
    os.remove(temp_file)
    if any(name.endswith(('.ps')) for name in os.listdir(sys.path[0])):
        r = os.system('/usr/bin/find ./ -type f -name "*.ps" | xargs /usr/bin/rm')

    command_line = env_path+"/bin/_cai -seqall %s -cfile Ehuman.cut -outfile %s" % (fasta_path, temp_file)
    os.system(command_line)
    cai_dict = extract_cai(temp_file)
    os.remove(temp_file)

    command_line = env_path+"/bin/_chips -seqall %s -outfile %s -nosum" % (fasta_path, temp_file)
    os.system(command_line)
    enc_dict = extract_enc(temp_file)
    os.remove(temp_file)

    metrics_result = merge_func(['GC', 'MFE', 'CAI', 'ENC'], gc_dict, mfe_dict, cai_dict, enc_dict)
    seq_name = list(gc_dict.keys())[0].split('_')[0]
    save_path = args.output_path
    metrics_result.to_csv(save_path,index=False,header=True)