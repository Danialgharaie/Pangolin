#!/usr/bin/env python3 

import argparse
import tempfile
import gffutils
import pandas as pd
import pyfastx
import pickle
import time
import pysam
import os
import shelve
import sys
import logging
import shutil

from pkg_resources import resource_filename

#from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue, Process


from pangolin.model import *

IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def one_hot_encode(seq, strand):
    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')
    if strand == '+':
        seq = np.asarray(list(map(int, list(seq))))
    elif strand == '-':
        seq = np.asarray(list(map(int, list(seq[::-1]))))
        seq = (5 - seq) % 5  # Reverse complement
    return IN_MAP[seq.astype('int8')]

    

def compute_scores_batch(batch_nr, ref_seqs, alt_seqs, strands, d, models, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_losses = []
    all_gains = []
    tensor_size = f"{ref_seqs[0].shape[-1]}|{alt_seqs[0].shape[-1]}"
    log.debug(f"Working on batch {batch_nr} with {len(strands)} variants of tensor shapes: {tensor_size}")
    for i in range(0, len(ref_seqs), batch_size):
        # Convert list of tensors to single tensor
        batch_ref = torch.stack(ref_seqs[i:i+batch_size]).to(device)
        batch_alt = torch.stack(alt_seqs[i:i+batch_size]).to(device)
        batch_strands = strands[i:i+batch_size]
        
        # Reshape to [batch_size, channels, length]
        batch_ref = batch_ref.view(batch_ref.size(0), batch_ref.size(2), batch_ref.size(3))
        batch_alt = batch_alt.view(batch_alt.size(0), batch_alt.size(2), batch_alt.size(3))
        
        pangolin_scores = []
        with torch.no_grad():
            for j in range(4):
                score = []
                for model in models[3*j:3*j+3]:
                    ref_pred = model(batch_ref)
                    alt_pred = model(batch_alt)
                    
                    ref = ref_pred[:, [1,4,7,10][j], :]
                    alt = alt_pred[:, [1,4,7,10][j], :]
                    
                    # Handle negative strands
                    neg_mask = [idx for idx, s in enumerate(batch_strands) if s == '-']
                    if neg_mask:
                        ref[neg_mask] = torch.flip(ref[neg_mask], dims=[1])
                        alt[neg_mask] = torch.flip(alt[neg_mask], dims=[1])
                    
                    # Handle length differences before calculating scores
                    l = 2*d+1
                    ndiff = abs(ref.shape[-1] - alt.shape[-1])
                    if ref.shape[-1] > alt.shape[-1]:
                        pad = torch.zeros(ref.shape[0], ndiff, device=device)
                        alt = torch.cat([alt[:, :l//2+1], pad, alt[:, l//2+1:]], dim=1)
                    elif ref.shape[-1] < alt.shape[-1]:
                        max_vals = alt[:, l//2:l//2+ndiff+1].max(dim=1, keepdim=True)[0]
                        alt = torch.cat([alt[:, :l//2], max_vals, alt[:, l//2+ndiff+1:]], dim=1)
                    
                    score.append(alt - ref)

                pangolin_scores.append(torch.stack(score).mean(dim=0))
        
        # Stack scores along a new dimension
        pangolin = torch.stack(pangolin_scores) 
        # Compute min and max scores across models (dim 0)
        batch_loss, _ = pangolin.min(dim=0)  
        batch_gain, _ = pangolin.max(dim=0)  
       
        # Move results to CPU and convert to lists of lists
        batch_loss_cpu = batch_loss.cpu() #.numpy()  # Shape: [batch_size, 101]
        batch_gain_cpu = batch_gain.cpu() #.numpy()  # Shape: [batch_size, 101]
        
        # keep cpu work as limited as possible here. reformat in the writer process
        all_losses.append(batch_loss_cpu)
        all_gains.append(batch_gain_cpu)
    
    return all_losses, all_gains


def get_genes(chr, pos, gtf):
    genes = gtf.region((chr, pos-1, pos-1), featuretype="gene")
    genes_pos, genes_neg = {}, {}

    for gene in genes:
        if gene[3] > pos or gene[4] < pos:
            continue
        gene_id = gene["gene_id"][0]
        exons = []
        for exon in gtf.children(gene, featuretype="exon"):
            exons.extend([exon[3], exon[4]])
        if gene[6] == '+':
            genes_pos[gene_id] = exons
        elif gene[6] == '-':
            genes_neg[gene_id] = exons

    return (genes_pos, genes_neg)



def prepare_variant(lnum, chr, pos, ref, alt, gtf, args, fasta, batches, skipped_variants):
    d = args.distance
    if len(set("ACGT").intersection(set(ref))) == 0 or len(set("ACGT").intersection(set(alt))) == 0 \
            or (len(ref) != 1 and len(alt) != 1 and len(ref) != len(alt)):
        log.warning("[Line %s]" % lnum, " skipping variant: Variant format not supported.")
        skipped_variants.add(f"{lnum}|{alt}")
        return skipped_variants, batches
    
    elif len(ref) > 2*d:
        log.warning(f"[Line {lnum}] skipping variant: Deletion too large")
        skipped_variants.add(f"{lnum}|{alt}")
        return skipped_variants, batches

    # try to make vcf chromosomes compatible with reference chromosomes
    if chr not in fasta.keys() and "chr"+chr in fasta.keys():
        chr = "chr"+chr
    elif chr not in fasta.keys() and chr[3:] in fasta.keys():
        chr = chr[3:]

    try:
        seq = fasta[chr][pos-5001-d:pos+len(ref)+4999+d].seq.upper()
    except Exception:
        skipped_variants.add(f"{lnum}|{alt}")
        return skipped_variants, batches    

    if seq[5000+d:5000+d+len(ref)] != ref:
        log.warning(f"[Line {lnum}] skipping variant: Mismatch between FASTA (ref base: {seq[5000+d:5000+d+len(ref)]}) and variant file (ref base: {ref}).")   
        skipped_variants.add(f"{lnum}|{alt}")
        return skipped_variants, batches

    ref_seq = seq
    alt_seq = seq[:5000+d] + alt + seq[5000+d+len(ref):]

    # get genes that intersect variant
    genes_pos, genes_neg = get_genes(chr, pos, gtf)
    if len(genes_pos)+len(genes_neg)==0:
        # no genes is not critical : keep on debug
        log.debug(f"[Line {lnum}] skipping variant {chr}:{pos} {ref}/{alt}: Variant not contained in a gene body. Do GTF/FASTA chromosome names match?")
        skipped_variants.add(f"{lnum}|{alt}")
        return skipped_variants, batches
    
    # encode
    if len(genes_pos) > 0:
        ref_seq_pos = one_hot_encode(ref_seq, '+').T
        ref_seq_pos = torch.from_numpy(np.expand_dims(ref_seq_pos, axis=0)).float()
        alt_seq_pos = one_hot_encode(alt_seq, '+').T
        alt_seq_pos = torch.from_numpy(np.expand_dims(alt_seq_pos, axis=0)).float()
        
    else:
        ref_seq_pos, alt_seq_pos = None, None
    if len(genes_neg) > 0:
        ref_seq_neg = one_hot_encode(ref_seq, '-').T
        ref_seq_neg = torch.from_numpy(np.expand_dims(ref_seq_neg, axis=0)).float()
        alt_seq_neg = one_hot_encode(alt_seq, '-').T
        alt_seq_neg = torch.from_numpy(np.expand_dims(alt_seq_neg, axis=0)).float()
        
    else:
        ref_seq_neg, alt_seq_neg = None, None
    
    # if all None : return false:
    if ref_seq_pos is None and ref_seq_neg is None:
        skipped_variants.add(f"{lnum}|{alt}")
        return skipped_variants, batches
    ## add to batches
    for (ref_allele, alt_allele, strand,genes) in [(ref_seq_pos, alt_seq_pos, '+', genes_pos), 
                                     (ref_seq_neg, alt_seq_neg, '-', genes_neg)
                                    ]:
        if ref_allele is None or alt_allele is None:
            continue
        # tensor size: 
        length = f"{ref_allele.shape[-1]}|{alt_allele.shape[-1]}"
        # add to batches
        if length not in batches:
            batches[length] = {'ref_seqs' : [], 'alt_seqs' : [], 'strands' : [], 'variant_keys' : [], 'genes' : []}
        # for scoring
        batches[length]['ref_seqs'].append(ref_allele)
        batches[length]['alt_seqs'].append(alt_allele)
        batches[length]['strands'].append(strand)
        batches[length]['variant_keys'].append(f"{lnum}|{alt}|{strand}") 
        # for post_prosessing
        batches[length]['genes'].append(genes)
    
    return skipped_variants, batches
        

def read_batches(batch_nr, tmpdir):
    # read
    with open(f"{tmpdir}/batch_{batch_nr}.genes.pickle", 'rb') as f:
        genes = pickle.load(f)
    with open(f"{tmpdir}/batch_{batch_nr}.skipped_in.pickle", 'rb') as f:
        skipped_variants = pickle.load(f)
    with open(f"{tmpdir}/batch_{batch_nr}.scores_out.pickle", 'rb') as f:
        ## these need to be reordered as follows: 
        # each entry in losses/gains is a gpu-batch of variants
        variant_keys, losses, gains = pickle.load(f) 
        # reformat
        all_losses = []
        for gpu_batch in losses:
            all_losses.extend([loss.tolist() for loss in gpu_batch.numpy()])
        all_gains = []
        for gpu_batch in gains:
            all_gains.extend([gain.tolist() for gain in gpu_batch.numpy()])
        losses = np.array(all_losses)
        gains = np.array(all_gains)
        scores = {variant_keys[i]: [losses[i], gains[i]] for i in range(len(variant_keys))}
    
    # and remove
    os.remove(f"{tmpdir}/batch_{batch_nr}.genes.pickle")
    os.remove(f"{tmpdir}/batch_{batch_nr}.skipped_in.pickle")
    os.remove(f"{tmpdir}/batch_{batch_nr}.scores_out.pickle")
    os.remove(f"{tmpdir}/batch_{batch_nr}.scores_in.pickle")

    return genes, skipped_variants, scores

def fill_shelve(tmpdir, queue):
    log.debug("Start writing to shelve")
    with shelve.open(f"{tmpdir}/variants.shelve") as sh:
        # read batches
        while True:
            batch_nr = queue.get()
            if batch_nr is None:
                log.debug(f"No more batches to process. close the shelve.")
                break
            # load the batch
            print (f"  => Loading batch {batch_nr}")
            genes, variants_skipped, variant_scores = read_batches(batch_nr, tmpdir)
            for variant_key in variants_skipped:
                # key is variant without +/-
                sh[f"{variant_key}|+"] = None
                sh[f"{variant_key}|-"] = None
            for idx in range(0,len(genes[0])):
                # key 
                if genes[0][idx] not in sh:
                    sh[genes[0][idx]] = {'genes': genes[1][idx]}
                else:
                    d = sh[genes[0][idx]]
                    d['genes'] = genes[1][idx]
                    sh[genes[0][idx]] = d
            for variant_key in variant_scores:
                # should match genes, but make sure
                if variant_key not in sh:
                    sh[variant_key] = {'loss': variant_scores[variant_key][0], 'gain': variant_scores[variant_key][1]}
                else: 
                    d = sh[variant_key]
                    d.update({'loss': variant_scores[variant_key][0], 'gain': variant_scores[variant_key][1]})
                    sh[variant_key] = d
            # write to disk
            sh.sync()  


def vcf_writer(queue, variants, args, tmpdir): # pos, ref_seq, alt_seq, genes_pos, genes_neg, models, args):
    d = args.distance
    cutoff = args.score_cutoff
    # variants are out of order (based on tensor size)
    #   1. create a shelve
    try:
        fill_shelve(tmpdir, queue)
    except Exception as e:
        log.error(f"Shelve creation failed: {repr(e)}")
        sys.exit(1)

    #   2. once all are ready => write VCF
    with pysam.VariantFile(variants) as variant_file, pysam.VariantFile(
            args.output_file+".vcf", "w", header=variant_file.header        
    ) as out_variant_file, shelve.open(f"{tmpdir}/variants.shelve") as sh:
        out_variant_file.header.add_meta(
            key="INFO",
            items=[
                ("ID", "Pangolin"),
                ("Number", "."),
                ("Type", "String"),
                (
                    "Description",
                    "Pangolin splice scores. Format: gene|pos:score_change|pos:score_change|warnings,..."
                    ),
                ] 
        )

        lnum = 0
        # count the number of header lines
        for line in open(variants, 'r'):
            lnum += 1
            if line[0] != '#':
                break
        
        # start processing:
        for idx, variant_record in enumerate(variant_file):
            variant_record.translate(out_variant_file.header)
            alt = str(variant_record.alts[0])
            pos = int(variant_record.pos)
            variant_key = f"{lnum+idx}|{alt}"
            
            # skipped variant
            if f"{variant_key}|+" not in sh and f"{variant_key}|-" not in sh:
                out_variant_file.write(variant_record)
                continue
            # get the scores
            if f"{variant_key}|+" in sh and sh[f"{variant_key}|+"] is not None:
                # get the scores
                loss_pos, gain_pos = sh[f"{variant_key}|+"]['loss'], sh[f"{variant_key}|+"]['gain']
                genes_pos = sh[f"{variant_key}|+"]['genes']
            else:
                loss_pos, gain_pos, genes_pos = None, None, None
            if f"{variant_key}|-" in sh and sh[f"{variant_key}|-"] is not None:
                loss_neg, gain_neg = sh[f"{variant_key}|-"]['loss'], sh[f"{variant_key}|-"]['gain']
                genes_neg = sh[f"{variant_key}|-"]['genes']
            else:
                loss_neg, gain_neg, genes_neg = None, None, None
            
            # reformat for vcf
            scores_list = []
            for (genes, loss, gain) in (
                (genes_pos,loss_pos,gain_pos),(genes_neg,loss_neg,gain_neg)
            ):
                if loss is None or gain is None or len(genes) == 0:
                    continue
                # Emit a bundle of scores/warnings per gene; join them all later
                for gene, positions in genes.items():
                    per_gene_scores = []
                    warnings = "Warnings:"
                    positions = np.array(positions)
                    positions = positions - (pos - d)
                    # apply masking
                    if args.mask == "True" and len(positions) != 0:
                        positions_filt = positions[(positions>=0) & (positions<len(loss))]
                        # set splice gain at annotated sites to 0
                        gain[positions_filt] = np.minimum(gain[positions_filt], 0)
                        # set splice loss at unannotated sites to 0
                        not_positions = ~np.isin(np.arange(len(loss)), positions_filt)
                        loss[not_positions] = np.maximum(loss[not_positions], 0)

                    elif args.mask == "True":
                        warnings += "NoAnnotatedSitesToMaskForThisGene"
                        loss[:] = np.maximum(loss[:], 0)

                    if args.score_exons == "True":
                        scores1 = [gene + '_sites1']
                        scores2 = [gene + '_sites2']

                        for i in range(len(positions)//2):
                            p1, p2 = positions[2*i], positions[2*i+1]
                            if p1<0 or p1>=len(loss):
                                s1 = "NA"
                            else:
                                s1 = [loss[p1],gain[p1]]
                                s1 = round(s1[np.argmax(np.abs(s1))],2)
                            if p2<0 or p2>=len(loss):
                                s2 = "NA"
                            else:
                                s2 = [loss[p2],gain[p2]]
                                s2 = round(s2[np.argmax(np.abs(s2))],2)
                            if s1 == "NA" and s2 == "NA":
                                continue
                            scores1.append(f"{p1-d}:{s1}")
                            scores2.append(f"{p2-d}:{s2}")
                        per_gene_scores += scores1 + scores2

                    elif cutoff != None:
                        per_gene_scores.append(gene)
                        l, g = np.where(loss<=-cutoff)[0], np.where(gain>=cutoff)[0]
                        for p, s in zip(np.concatenate([g-d,l-d]), np.concatenate([gain[g],loss[l]])):
                            per_gene_scores.append(f"{p}:{round(s,2)}")

                    else:
                        per_gene_scores.append(gene)
                        l, g = np.argmin(loss), np.argmax(gain),
                        gain_str = f"{g-d}:{round(gain[g],2)}"
                        loss_str = f"{l-d}:{round(loss[l],2)}"
                        per_gene_scores += [gain_str, loss_str]

                    per_gene_scores.append(warnings)
                    scores_list.append('|'.join(per_gene_scores))

            # write to vcf
            variant_record.info["Pangolin"] = ",".join(scores_list)
            out_variant_file.write(variant_record)

    # remove the shelve
    #os.remove(f"{tmpdir}/variants.shelve")

def pickle_batches(batches, skipped_variants, batch_nr, tmpdir, queue, args, all=False):
    for length in batches:
        # can be equal or +1 to variant_batchsize
        if all == True or len(batches[length]['strands']) >= args.variant_batchsize:
            # genes info
            with open(f"{tmpdir}/batch_{batch_nr}.genes.pickle", 'wb') as f:
                pickle.dump((batches[length]['variant_keys'], batches[length]['genes']), f)
            # skipped variants so far
            with open(f"{tmpdir}/batch_{batch_nr}.skipped_in.pickle", 'wb') as f:
                pickle.dump(skipped_variants, f)
            # data to run scoring
            with open(f"{tmpdir}/batch_{batch_nr}.scores_in.pickle", 'wb') as f:
                pickle.dump((batches[length]['ref_seqs'], batches[length]['alt_seqs'], batches[length]['strands'], batches[length]['variant_keys']), f)
            # reset for next batch
            batches[length] = {'ref_seqs' : [], 'alt_seqs' : [], 'strands' : [], 'variant_keys' : [], 'genes' : []}
            skipped_variants = set()
            # add to queue
            queue.put(batch_nr)
            # increase batch nr
            batch_nr += 1
    return batch_nr, batches, skipped_variants

def vcf_reader(variants, queue, gtf, args, tmpdir ):
    # open fasta file
    fasta = pyfastx.Fasta(args.reference_file)
    # batch nr for tracking order
    batch_nr = 0

    lnum = 0
    # count the number of header lines
    for line in open(variants, 'r'):
        lnum += 1
        if line[0] != '#':
            break
    
    #line numbers 
    skipped_variants = set()
    # group variants for scoring by tensor size
    batches = dict()
    with pysam.VariantFile(variants) as variant_file:
        for i, variant_record in enumerate(variant_file):
            # validate variant
            assert variant_record.ref, f"Empty REF field in variant record {variant_record}"
            assert variant_record.alts, f"Empty ALT field in variant record {variant_record}"
            skipped_variants, batches = prepare_variant(
                    lnum + i,
                    str(variant_record.contig),
                    int(variant_record.pos),
                    str(variant_record.ref).upper(),
                    str(variant_record.alts[0]).upper(),
                    gtf,
                    args,
                    fasta,
                    batches,
                    skipped_variants
                    )
            # pickle 
            batch_nr, batches, skipped_variants = pickle_batches(batches, skipped_variants, batch_nr, tmpdir, queue, args)

        # pickle remaining batches       
        batch_nr, batches, skipped_variants = pickle_batches(batches, skipped_variants, batch_nr, tmpdir, queue, args, all=True)
    
    # send the sentinel value to indicate the end of the queue
    queue.put(None)
    

def csv_reader(variants, queue, gtf, args, tmpdir):
     # open fasta file
    fasta = pyfastx.Fasta(args.reference_file)
    # batch nr for tracking order
    batch_nr = 0

    col_ids = args.column_ids.split(',')
    variants = pd.read_csv(variants, header=0)
    
    #line numbers 
    skipped_variants = set()
    # group variants for scoring by tensor size
    batches = dict()

    for lnum, variant in variants.iterrows():
        chr, pos, ref, alt = variant[col_ids]
        ref, alt = ref.upper(), alt.upper()
        # validate variant
        assert ref, f"Empty REF field in variant record {lnum}"
        assert alt, f"Empty ALT field in variant record {lnum}"
        skipped_variants, batches = prepare_variant(
                lnum ,
                str(chr),
                int(pos),
                ref.upper(),
                alt.upper(),
                gtf,
                args,
                fasta,
                batches,
                skipped_variants
                )
        batch_nr, batches, skipped_variants = pickle_batches(batches, skipped_variants, batch_nr, tmpdir, queue, args)
        
    # pickle all remaining batches
    batch_nr, batches, skipped_variants = pickle_batches(batches, skipped_variants, batch_nr, tmpdir, queue, args, all=True)

    # send the sentinel value to indicate the end of the queue
    queue.put(None)
    

# minimal function to load the models and score the variant batches
def scoring(scoring_queue, writing_queue, args, tmpdir):
    # load models
    d = args.distance
    models = []
    for i in [0,2,4,6]:
        for j in range(1,4):
            model = Pangolin(L, W, AR)
            if torch.cuda.is_available():
                model.cuda()
                weights = torch.load(resource_filename(__name__,"models/final.%s.%s.3.v2" % (j, i)))
            else:
                weights = torch.load(resource_filename(__name__,"models/final.%s.%s.3.v2" % (j, i)), map_location=torch.device('cpu'))
            model.load_state_dict(weights)
            model.eval()
            models.append(model)
    
    # process the queue
    while True:
        item = scoring_queue.get()
        if item is None:
            log.debug("found sentinel on scoring queue. Close worker")
            break
        with open(f"{tmpdir}/batch_{item}.scores_in.pickle", 'rb') as f:
            ref_seqs, alt_seqs, strands, variant_keys = pickle.load(f)
        
        # score batch variants
        batch_time_start = time.time()
        losses, gains = compute_scores_batch(item, ref_seqs, alt_seqs, strands, d, models, batch_size=args.tensor_batchsize)
                
        # pickle
        batch_time_end = time.time()
        print(f"Scored {len(ref_seqs)} variants in {int(batch_time_end - batch_time_start)} seconds : {int(len(ref_seqs)/((batch_time_end-batch_time_start)/3600))} variants/hour")
        with open(f"{tmpdir}/batch_{item}.scores_out.pickle", 'wb') as f:
            pickle.dump((variant_keys, losses, gains), f)

        writing_queue.put(item)

    # add end of queue signal
    writing_queue.put(None)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant_file", help="VCF or CSV file with a header (see COLUMN_IDS option).")
    parser.add_argument("reference_file", help="FASTA file containing a reference genome sequence.")
    parser.add_argument("annotation_file", help="gffutils database file. Can be generated using create_db.py.")
    parser.add_argument("output_file", help="Prefix for output file. Will be a VCF/CSV if variant_file is VCF/CSV.")
    parser.add_argument("-c", "--column_ids", default="CHROM,POS,REF,ALT", help="(If variant_file is a CSV) Column IDs for: chromosome, variant position, reference bases, and alternative bases. "
                                                                                "Separate IDs by commas. (Default: CHROM,POS,REF,ALT)")
    parser.add_argument("-m", "--mask", default="True", choices=["False","True"], help="If True, splice gains (increases in score) at annotated splice sites and splice losses (decreases in score) at unannotated splice sites will be set to 0. (Default: True)")
    parser.add_argument("-s", "--score_cutoff", type=float, help="Output all sites with absolute predicted change in score >= cutoff, instead of only the maximum loss/gain sites.")
    parser.add_argument("-d", "--distance", type=int, default=50, help="Number of bases on either side of the variant for which splice scores should be calculated. (Default: 50)")
    parser.add_argument("--score_exons", default="False", choices=["False","True"], help="Output changes in score for both splice sites of annotated exons, as long as one splice site is within the considered range (specified by -d). Output will be: gene|site1_pos:score|site2_pos:score|...")
    parser.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level. (Default: INFO)")
    parser.add_argument("--tmpdir", help="Location to create temporary directory for storing intermediate files.", default=tempfile.gettempdir())
    parser.add_argument("--variant_batchsize", type=int, default=1000, help="Number of variants to score in a single CPU batch. (Default: 1000)")
    parser.add_argument("--tensor_batchsize", type=int, default=128, help="Number of variants to process in a single GPU batch. (Default: 128)")
    args = parser.parse_args()

    # logging
    logging.basicConfig(level=args.loglevel,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    global log
    log = logging.getLogger(__name__)

    # start time:
    log.info("Starting Pangolin")
    startTime = time.time()

    variants = args.variant_file
    gtf = args.annotation_file
    try:
        gtf = gffutils.FeatureDB(gtf)
    except Exception as e:
        log.error(f"Annotation_file could not be opened ({repr(e)}). Is it a gffutils database file?")
        sys.exit(1)

    if torch.cuda.is_available():
        log.info("Using GPU")
    else:
        log.info("Using CPU")
    
    # tmp dir to pickle results
    tmpdir = tempfile.mkdtemp(dir=args.tmpdir)
    log.debug(f"Temporary directory for storing intermediate files: {tmpdir}")
    # create queues:
    scoring_queue = Queue(maxsize=25)
    writing_queue = Queue(maxsize=25)
    # create variant reader process:
    if variants.endswith(".vcf"):
        # process 1 : vcf => batch-pickles
        reader = Process(target=vcf_reader, kwargs={"queue" : scoring_queue, "variants": variants, "gtf": gtf, "args": args, 'tmpdir': tmpdir})
        reader.start()
        # process 3 : read batches into shelve => write VCF
        writer = Process(target=vcf_writer, kwargs={"queue" : writing_queue, "variants": variants, "args": args, 'tmpdir': tmpdir})
        writer.start()

    elif variants.endswith(".csv"):
        reader = Process(target=csv_reader, kwargs={"queue" : scoring_queue, "variants": variants, "gtf": gtf, "args": args, 'tmpdir': tmpdir})
        reader.start()
        # process 3 : read batches into shelve => write VCF
        writer = Process(target=csv_writer, kwargs={"queue" : writing_queue, "variants": variants, "args": args, 'tmpdir': tmpdir})
        writer.start()

    else:
        log.error("Variant_file needs to be a CSV or VCF.")

    # score the variants (pickle => gpu => pickle)
    #  todo : if this can be subprocessed, we can use multiple gpus. (torch.multiprocessing ?) 
    try:
        scoring(scoring_queue=scoring_queue,writing_queue=writing_queue,args=args,tmpdir=tmpdir)
    except Exception as e:
        log.error(f"Scoring process failed: {repr(e)}")
        sys.exit(1)

    # join the reader
    reader.join()
    # close first queue
    scoring_queue.close()

    # join the writer
    writer.join()
    # close second queue
    writing_queue.close()


    executionTime = (time.time() - startTime)
    # format exec time in hours:minutes:sec
    log.info(f"Execution time: {int(executionTime//3600)}h:{int((executionTime%3600)//60)}m:{int(executionTime%60)}s")

    # remove temp folder and contents
    shutil.rmtree(tmpdir)
    

if __name__ == '__main__':
    main()
