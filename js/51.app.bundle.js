(window.webpackJsonp=window.webpackJsonp||[]).push([[51,81],{371:function(e,n,a){"use strict";function t(e){function n(e,n){return"___"+e.toUpperCase()+n+"___"}var a;a=e,Object.defineProperties(a.languages["markup-templating"]={},{buildPlaceholders:{value:function(e,t,o,r){var s;e.language===t&&(s=e.tokenStack=[],e.code=e.code.replace(o,(function(a){if("function"==typeof r&&!r(a))return a;for(var o,i=s.length;-1!==e.code.indexOf(o=n(t,i));)++i;return s[i]=a,o})),e.grammar=a.languages.markup)}},tokenizePlaceholders:{value:function(e,t){var o,r;e.language===t&&e.tokenStack&&(e.grammar=a.languages[t],o=0,r=Object.keys(e.tokenStack),function s(i){for(var l=0;l<i.length&&!(o>=r.length);l++){var u,c,p,g,d,k,f,b,m,h=i[l];"string"==typeof h||h.content&&"string"==typeof h.content?(u=r[o],c=e.tokenStack[u],p="string"==typeof h?h:h.content,g=n(t,u),-1<(d=p.indexOf(g))&&(++o,k=p.substring(0,d),f=new a.Token(t,a.tokenize(c,e.grammar),"language-"+t,c),b=p.substring(d+g.length),m=[],k&&m.push.apply(m,s([k])),m.push(f),b&&m.push.apply(m,s([b])),"string"==typeof h?i.splice.apply(i,[l,1].concat(m)):h.content=m)):h.content&&s(h.content)}return i}(e.tokens))}}})}(e.exports=t).displayName="markupTemplating",t.aliases=[]},463:function(e,n,a){"use strict";var t=a(371);function o(e){var n;e.register(t),(n=e).languages.handlebars={comment:/\{\{![\s\S]*?\}\}/,delimiter:{pattern:/^\{\{\{?|\}\}\}?$/i,alias:"punctuation"},string:/(["'])(?:\\.|(?!\1)[^\\\r\n])*\1/,number:/\b0x[\dA-Fa-f]+\b|(?:\b\d+\.?\d*|\B\.\d+)(?:[Ee][+-]?\d+)?/,boolean:/\b(?:true|false)\b/,block:{pattern:/^(\s*~?\s*)[#\/]\S+?(?=\s*~?\s*$|\s)/i,lookbehind:!0,alias:"keyword"},brackets:{pattern:/\[[^\]]+\]/,inside:{punctuation:/\[|\]/,variable:/[\s\S]+/}},punctuation:/[!"#%&'()*+,.\/;<=>@\[\\\]^`{|}~]/,variable:/[^!"#%&'()*+,.\/;<=>@\[\\\]^`{|}~\s]+/},n.hooks.add("before-tokenize",(function(e){n.languages["markup-templating"].buildPlaceholders(e,"handlebars",/\{\{\{[\s\S]+?\}\}\}|\{\{[\s\S]+?\}\}/g)})),n.hooks.add("after-tokenize",(function(e){n.languages["markup-templating"].tokenizePlaceholders(e,"handlebars")}))}(e.exports=o).displayName="handlebars",o.aliases=[]}}]);