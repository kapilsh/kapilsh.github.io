(window.webpackJsonp=window.webpackJsonp||[]).push([[81],{385:function(e,n,t){"use strict";function a(e){function n(e,n){return"___"+e.toUpperCase()+n+"___"}var t;t=e,Object.defineProperties(t.languages["markup-templating"]={},{buildPlaceholders:{value:function(e,a,o,r){var c;e.language===a&&(c=e.tokenStack=[],e.code=e.code.replace(o,(function(t){if("function"==typeof r&&!r(t))return t;for(var o,p=c.length;-1!==e.code.indexOf(o=n(a,p));)++p;return c[p]=t,o})),e.grammar=t.languages.markup)}},tokenizePlaceholders:{value:function(e,a){var o,r;e.language===a&&e.tokenStack&&(e.grammar=t.languages[a],o=0,r=Object.keys(e.tokenStack),function c(p){for(var u=0;u<p.length&&!(o>=r.length);u++){var i,s,g,l,f,k,d,m,h,y=p[u];"string"==typeof y||y.content&&"string"==typeof y.content?(i=r[o],s=e.tokenStack[i],g="string"==typeof y?y:y.content,l=n(a,i),-1<(f=g.indexOf(l))&&(++o,k=g.substring(0,f),d=new t.Token(a,t.tokenize(s,e.grammar),"language-"+a,s),m=g.substring(f+l.length),h=[],k&&h.push.apply(h,c([k])),h.push(d),m&&h.push.apply(h,c([m])),"string"==typeof y?p.splice.apply(p,[u,1].concat(h)):y.content=h)):y.content&&c(y.content)}return p}(e.tokens))}}})}(e.exports=a).displayName="markupTemplating",a.aliases=[]}}]);