(window.webpackJsonp=window.webpackJsonp||[]).push([[137,81],{389:function(e,t,n){"use strict";function a(e){function t(e,t){return"___"+e.toUpperCase()+t+"___"}var n;n=e,Object.defineProperties(n.languages["markup-templating"]={},{buildPlaceholders:{value:function(e,a,r,o){var s;e.language===a&&(s=e.tokenStack=[],e.code=e.code.replace(r,(function(n){if("function"==typeof o&&!o(n))return n;for(var r,i=s.length;-1!==e.code.indexOf(r=t(a,i));)++i;return s[i]=n,r})),e.grammar=n.languages.markup)}},tokenizePlaceholders:{value:function(e,a){var r,o;e.language===a&&e.tokenStack&&(e.grammar=n.languages[a],r=0,o=Object.keys(e.tokenStack),function s(i){for(var g=0;g<i.length&&!(r>=o.length);g++){var u,l,p,c,d,f,k,E,S,m=i[g];"string"==typeof m||m.content&&"string"==typeof m.content?(u=o[r],l=e.tokenStack[u],p="string"==typeof m?m:m.content,c=t(a,u),-1<(d=p.indexOf(c))&&(++r,f=p.substring(0,d),k=new n.Token(a,n.tokenize(l,e.grammar),"language-"+a,l),E=p.substring(d+c.length),S=[],f&&S.push.apply(S,s([f])),S.push(k),E&&S.push.apply(S,s([E])),"string"==typeof m?i.splice.apply(i,[g,1].concat(S)):m.content=S)):m.content&&s(m.content)}return i}(e.tokens))}}})}(e.exports=a).displayName="markupTemplating",a.aliases=[]},559:function(e,t,n){"use strict";var a=n(389);function r(e){var t;e.register(a),(t=e).languages.tt2=t.languages.extend("clike",{comment:/#.*|\[%#[\s\S]*?%\]/,keyword:/\b(?:BLOCK|CALL|CASE|CATCH|CLEAR|DEBUG|DEFAULT|ELSE|ELSIF|END|FILTER|FINAL|FOREACH|GET|IF|IN|INCLUDE|INSERT|LAST|MACRO|META|NEXT|PERL|PROCESS|RAWPERL|RETURN|SET|STOP|TAGS|THROW|TRY|SWITCH|UNLESS|USE|WHILE|WRAPPER)\b/,punctuation:/[[\]{},()]/}),t.languages.insertBefore("tt2","number",{operator:/=[>=]?|!=?|<=?|>=?|&&|\|\|?|\b(?:and|or|not)\b/,variable:{pattern:/[a-z]\w*(?:\s*\.\s*(?:\d+|\$?[a-z]\w*))*/i}}),t.languages.insertBefore("tt2","keyword",{delimiter:{pattern:/^(?:\[%|%%)-?|-?%]$/,alias:"punctuation"}}),t.languages.insertBefore("tt2","string",{"single-quoted-string":{pattern:/'[^\\']*(?:\\[\s\S][^\\']*)*'/,greedy:!0,alias:"string"},"double-quoted-string":{pattern:/"[^\\"]*(?:\\[\s\S][^\\"]*)*"/,greedy:!0,alias:"string",inside:{variable:{pattern:/\$(?:[a-z]\w*(?:\.(?:\d+|\$?[a-z]\w*))*)/i}}}}),delete t.languages.tt2.string,t.hooks.add("before-tokenize",(function(e){t.languages["markup-templating"].buildPlaceholders(e,"tt2",/\[%[\s\S]+?%\]/g)})),t.hooks.add("after-tokenize",(function(e){t.languages["markup-templating"].tokenizePlaceholders(e,"tt2")}))}(e.exports=r).displayName="tt2",r.aliases=[]}}]);