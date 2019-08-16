dbodydict = dict() # dtitle, dbody

fdbody = open("sample_alldoc_fake_body.dict")
while 1:
    line = fdbody.readline()
    if not line:
        break
    tks = line.strip().split(' ')
    dbodydict[tks[0]]=tks[1]

fclean = open("sample_test_with_ids_cqexp.cleaned")
fexp = open("sample_test_with_ids_cqbody.cleaned",'w')

while 1:
    line = fclean.readline()
    if not line:
        break
    tks = line.strip().split(' ')

    q = tks[0]
    d = tks[1]
    qid = tks[2]
    dsogouid = tks[3]
    dcq = tks[4]

    dbody = '0'
    if d in dbodydict:
        dbody = dbodydict[d]

    out = str(q)+' '+str(d)+' '+str(qid)+' '+str(dsogouid)+' '+str(dcq)+' '+str(dbody)+'\n'

    fexp.write(out)


