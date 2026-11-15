import json,sys
d=open('data.json').read(); e=open('engine.js').read(); p=open('page.html').read()
out=p.replace('/*__ENGINE__*/',e).replace('/*__DATA__*/',d)
open('ev-charging-explainer.html','w').write(out); print(len(out)//1024,'KB')
