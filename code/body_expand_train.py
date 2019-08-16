dbodydict = dict() # dtitle, dbody

fdbody = open("sample_alldoc_fake_body.dict")
while 1:
    line = fdbody.readline()
    if not line:
        break
    tks = line.strip().split(' ')
    dbodydict[tks[0]]=tks[1]

fclean = open("sample_valid_cqexp.cleaned")
fexp = open("sample_valid_cqbody.cleaned",'w')

while 1:
    line = fclean.readline()
    if not line:
        break
    tks = line.strip().split(' ')

    q = tks[0]
    dp = tks[1]
    dn = tks[2]
    dpcq = tks[3]
    dncq = tks[4]

    dpbody = '0'
    dnbody = '0'
    if dp in dbodydict:
        dpbody = dbodydict[dp]
    if dn in dbodydict:
        dnbody = dbodydict[dn]

    out = str(q)+' '+str(dp)+' '+str(dn)+' '+str(dpcq)+' '+str(dncq)+' '+str(dpbody)+' '+str(dnbody)+'\n'
    fexp.write(out)


