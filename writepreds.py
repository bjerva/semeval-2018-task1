import semeval2018


def printPredsToFileReg(infile, outfile, res, infileenc="utf-8"):
    outf = open(outfile, 'w', encoding=infileenc)
    with open(infile, encoding=infileenc, mode='r') as f:
        outf.write(f.readline())
        for i, line in f:        
            outl = line.strip("\n").split("\t")
            outl[3] = str(res[i])
            outf.write("\t".join(outl) + '\n')