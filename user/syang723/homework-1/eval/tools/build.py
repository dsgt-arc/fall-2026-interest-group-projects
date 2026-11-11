import json,re,sys,sqlite3
sys.path.insert(0,sys.argv[2]); from hard_spec import Q
d=json.load(open(sys.argv[1])); out=sys.argv[3]
db=sqlite3.connect(sys.argv[4]); track={i:t.split(':')[0] for i,t in db.execute('select id,track from documents')}
norm={p:[re.sub(r'\s+',' ',t) for t in pages] for p,pages in d.items()}
# "track": keep all anchor hits in the intended paper's track; list: explicit pages; default: intended paper only (all its hits)
OVERRIDE={
 "bioasq-1":[("paper18",5),("paper18",12)],
 "exist-2":[("paper159",2),("paper159",5)],
 "exist-4":"track","joker-1":[("paper305",5)],"joker-2":[("paper305",3),("paper307",16),("paper307",17),("paper315",2)],"joker-3":"track","lifeclef-1":"track","lifeclef-2":"track","erisk-1":"track",
}
cases=[]
for qid,cat,q,pid,pg,anch,exp in Q:
    hits=[(p,i+1) for p,pages in norm.items() for i,t in enumerate(pages) if all(re.search(a,t,re.I) for a in anch)]
    assert (pid,pg) in hits,(qid,hits)
    o=OVERRIDE.get(qid)
    if o=="track": gold=[h for h in hits if track[h[0]]==track[pid]]
    elif o: gold=o
    else: gold=[h for h in hits if h[0]==pid]
    by={}
    for p,n in gold: by.setdefault(p,[]).append(n)
    cases.append({"id":f"hard-{qid}","category":cat,"question":q,
        "gold_sources":[{"paper":p,"pages":sorted(v)} for p,v in sorted(by.items(),key=lambda kv:(kv[0]!=pid,kv[0]))],
        "expected_answer":exp})
    print(qid, [(g['paper'],g['pages']) for g in cases[-1]['gold_sources']])
json.dump(cases,open(out,'w'),indent=2,ensure_ascii=False); open(out,'a').write('\n')
print(len(cases),'questions,',sum(len(c['gold_sources'])>1 for c in cases),'with multiple papers')
