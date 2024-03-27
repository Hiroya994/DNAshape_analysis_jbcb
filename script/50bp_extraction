## motif_extraction

# this code is written by python3

def Ao_motif_extract(seq_pass, upstream, all_genome, all_genes, motif_list, extract_len):
	
	up_file  	= seq_pass + upstream
	genome_file = seq_pass + all_genome
	genes_file	= seq_pass + all_genes
	
	motif_len	= len(motif_list[1])
	max_len		= extract_len*2 + motif_len
	
	##### import library #####
	
	import pandas as pd
	from Bio import SeqIO
	import re
	
	##### definition of find_double_max #####
	
	def find_double_max(seq, motif_list):
	
	## -----------------------------------------------
	# detect the multiple binding motif
	# -> get the max number of all position of motifs
	## -----------------------------------------------
	
	# seq	= detected sequences
	# motif = binding motifs you want to detect
	
		index_L = []
		
		for motif in motif_list:
		
			if seq.count(motif) == 0:
				index_L.append(-1)

			elif seq.count(motif) == 1:
				x = seq.find(motif)
				index_L.append(x)

			else:
				pos = seq.find(motif)
			
				for i in range(1,(seq.count(motif) + 1)):
					if i == 1:
						index_L.append(pos)
				
					else:
						st = pos + 1
						pos = seq.find(motif, st)
						index_L.append(pos)

		return max(index_L)
	
	##### empty list for making fasta file #####
	
	id_list = []
	extract_list = []
	
	##### read fasta by each and search motifs #####
	
	for seq_record in SeqIO.parse(up_file, "fasta"):

		id_part = seq_record.id
		info_part = str(seq_record.description)
		upstream_part = str(seq_record.seq)
		pos = find_double_max(upstream_part, motif_list)

		if pos == -1: # if there is no motif, pass
			pass

		elif pos < extract_len: # if the motif is in the upstream part
			up_extract = upstream_part[:pos + extract_len + motif_len]
			x = re.split(":", info_part)
			y = re.split("-", x[3])
			z = re.split(" ", y[1])
			chr = x[2]
			id_list.append(seq_record.id)

			if z[0].endswith('W') == True:
				end_position = int(y[0])+(len(up_extract)-1)
				start_position = end_position - max_len

				for chr_seq in SeqIO.parse(genome_file, "fasta"):
					if chr_seq.id == chr:
						extract_list.append(str(chr_seq.seq[start_position:end_position]))

					else:
						pass

			else:
				start_position = int(z[0].strip('C')) - (len(up_extract))
				end_position = start_position + max_len

				for chr_seq in SeqIO.parse(genome_file, "fasta"):
					if chr_seq.id == chr:
						comp_seq = str(chr_seq.seq[start_position:end_position])
						true_seq = comp_seq[::-1].translate(str.maketrans("AGCT", "TCGA"))
						extract_list.append(true_seq)

					else:
						pass

		elif len(upstream_part)-pos < (extract_len + motif_len): # if the motif is in the downstream part
			down_extract = upstream_part[pos-(extract_len-1):]

			for seq_record_2 in SeqIO.parse(genes_file, "fasta"): # read the original seq again and retry the detection 
				original_seq = str(seq_record_2.seq)

				if seq_record_2.id == id_part:
					additional_seq = original_seq[original_seq.find(down_extract)-1:(original_seq.find(down_extract)+(max_len-1))]
					id_list.append(seq_record.id)
					extract_list.append(additional_seq)

				else:
					pass

		else:
			extract_seq = upstream_part[(pos - extract_len):(pos + extract_len + motif_len)] # if the 50bp region was correctly cut
			id_list.append(seq_record.id)
			extract_list.append(extract_seq)

	ofile = open("motif_extract.fasta", "w")
	for i in range(len(id_list)):
		ofile.write("> " + id_list[i] + "\n" + extract_list[i] + "\n") # write to the new fasta file
	ofile.close()
