(window.webpackJsonp=window.webpackJsonp||[]).push([[100,81],{388:function(e,n,t){"use strict";function a(e){function n(e,n){return"___"+e.toUpperCase()+n+"___"}var t;t=e,Object.defineProperties(t.languages["markup-templating"]={},{buildPlaceholders:{value:function(e,a,i,r){var o;e.language===a&&(o=e.tokenStack=[],e.code=e.code.replace(i,(function(t){if("function"==typeof r&&!r(t))return t;for(var i,s=o.length;-1!==e.code.indexOf(i=n(a,s));)++s;return o[s]=t,i})),e.grammar=t.languages.markup)}},tokenizePlaceholders:{value:function(e,a){var i,r;e.language===a&&e.tokenStack&&(e.grammar=t.languages[a],i=0,r=Object.keys(e.tokenStack),function o(s){for(var l=0;l<s.length&&!(i>=r.length);l++){var p,c,u,g,d,f,h,m,k,b=s[l];"string"==typeof b||b.content&&"string"==typeof b.content?(p=r[i],c=e.tokenStack[p],u="string"==typeof b?b:b.content,g=n(a,p),-1<(d=u.indexOf(g))&&(++i,f=u.substring(0,d),h=new t.Token(a,t.tokenize(c,e.grammar),"language-"+a,c),m=u.substring(d+g.length),k=[],f&&k.push.apply(k,o([f])),k.push(h),m&&k.push.apply(k,o([m])),"string"==typeof b?s.splice.apply(s,[l,1].concat(k)):b.content=k)):b.content&&o(b.content)}return s}(e.tokens))}}})}(e.exports=a).displayName="markupTemplating",a.aliases=[]},407:function(e,n,t){"use strict";var a=t(388);function i(e){e.register(a),function(e){e.languages.php=e.languages.extend("clike",{keyword:/\b(?:__halt_compiler|abstract|and|array|as|break|callable|case|catch|class|clone|const|continue|declare|default|die|do|echo|else|elseif|empty|enddeclare|endfor|endforeach|endif|endswitch|endwhile|eval|exit|extends|final|finally|for|foreach|function|global|goto|if|implements|include|include_once|instanceof|insteadof|interface|isset|list|namespace|new|or|parent|print|private|protected|public|require|require_once|return|static|switch|throw|trait|try|unset|use|var|while|xor|yield)\b/i,boolean:{pattern:/\b(?:false|true)\b/i,alias:"constant"},constant:[/\b[A-Z_][A-Z0-9_]*\b/,/\b(?:null)\b/i],comment:{pattern:/(^|[^\\])(?:\/\*[\s\S]*?\*\/|\/\/.*)/,lookbehind:!0}}),e.languages.insertBefore("php","string",{"shell-comment":{pattern:/(^|[^\\])#.*/,lookbehind:!0,alias:"comment"}}),e.languages.insertBefore("php","comment",{delimiter:{pattern:/\?>$|^<\?(?:php(?=\s)|=)?/i,alias:"important"}}),e.languages.insertBefore("php","keyword",{variable:/\$+(?:\w+\b|(?={))/i,package:{pattern:/(\\|namespace\s+|use\s+)[\w\\]+/,lookbehind:!0,inside:{punctuation:/\\/}}}),e.languages.insertBefore("php","operator",{property:{pattern:/(->)[\w]+/,lookbehind:!0}});var n={pattern:/{\$(?:{(?:{[^{}]+}|[^{}]+)}|[^{}])+}|(^|[^\\{])\$+(?:\w+(?:\[.+?]|->\w+)*)/,lookbehind:!0,inside:{rest:e.languages.php}};e.languages.insertBefore("php","string",{"nowdoc-string":{pattern:/<<<'([^']+)'(?:\r\n?|\n)(?:.*(?:\r\n?|\n))*?\1;/,greedy:!0,alias:"string",inside:{delimiter:{pattern:/^<<<'[^']+'|[a-z_]\w*;$/i,alias:"symbol",inside:{punctuation:/^<<<'?|[';]$/}}}},"heredoc-string":{pattern:/<<<(?:"([^"]+)"(?:\r\n?|\n)(?:.*(?:\r\n?|\n))*?\1;|([a-z_]\w*)(?:\r\n?|\n)(?:.*(?:\r\n?|\n))*?\2;)/i,greedy:!0,alias:"string",inside:{delimiter:{pattern:/^<<<(?:"[^"]+"|[a-z_]\w*)|[a-z_]\w*;$/i,alias:"symbol",inside:{punctuation:/^<<<"?|[";]$/}},interpolation:n}},"single-quoted-string":{pattern:/'(?:\\[\s\S]|[^\\'])*'/,greedy:!0,alias:"string"},"double-quoted-string":{pattern:/"(?:\\[\s\S]|[^\\"])*"/,greedy:!0,alias:"string",inside:{interpolation:n}}}),delete e.languages.php.string,e.hooks.add("before-tokenize",(function(n){/<\?/.test(n.code)&&e.languages["markup-templating"].buildPlaceholders(n,"php",/<\?(?:[^"'/#]|\/(?![*/])|("|')(?:\\[\s\S]|(?!\1)[^\\])*\1|(?:\/\/|#)(?:[^?\n\r]|\?(?!>))*|\/\*[\s\S]*?(?:\*\/|$))*?(?:\?>|$)/gi)})),e.hooks.add("after-tokenize",(function(n){e.languages["markup-templating"].tokenizePlaceholders(n,"php")}))}(e)}(e.exports=i).displayName="php",i.aliases=[]}}]);