(window.webpackJsonp=window.webpackJsonp||[]).push([[136,67,139],{403:function(t,e,n){"use strict";function a(t){!function(t){var e=t.util.clone(t.languages.javascript);t.languages.jsx=t.languages.extend("markup",e),t.languages.jsx.tag.pattern=/<\/?(?:[\w.:-]+\s*(?:\s+(?:[\w.:-]+(?:=(?:("|')(?:\\[\s\S]|(?!\1)[^\\])*\1|[^\s{'">=]+|\{(?:\{(?:\{[^}]*\}|[^{}])*\}|[^{}])+\}))?|\{\.{3}[a-z_$][\w$]*(?:\.[a-z_$][\w$]*)*\}))*\s*\/?)?>/i,t.languages.jsx.tag.inside.tag.pattern=/^<\/?[^\s>\/]*/i,t.languages.jsx.tag.inside["attr-value"].pattern=/=(?!\{)(?:("|')(?:\\[\s\S]|(?!\1)[^\\])*\1|[^\s'">]+)/i,t.languages.jsx.tag.inside.tag.inside["class-name"]=/^[A-Z]\w*(?:\.[A-Z]\w*)*$/,t.languages.insertBefore("inside","attr-name",{spread:{pattern:/\{\.{3}[a-z_$][\w$]*(?:\.[a-z_$][\w$]*)*\}/,inside:{punctuation:/\.{3}|[{}.]/,"attr-value":/\w+/}}},t.languages.jsx.tag),t.languages.insertBefore("inside","attr-value",{script:{pattern:/=(\{(?:\{(?:\{[^}]*\}|[^}])*\}|[^}])+\})/i,inside:{"script-punctuation":{pattern:/^=(?={)/,alias:"punctuation"},rest:t.languages.jsx},alias:"language-javascript"}},t.languages.jsx.tag);var n=function(t){return t?"string"==typeof t?t:"string"==typeof t.content?t.content:t.content.map(n).join(""):""},a=function(e){for(var s=[],i=0;i<e.length;i++){var o,r=e[i],g=!1;"string"!=typeof r&&("tag"===r.type&&r.content[0]&&"tag"===r.content[0].type?"</"===r.content[0].content[0].content?0<s.length&&s[s.length-1].tagName===n(r.content[0].content[1])&&s.pop():"/>"===r.content[r.content.length-1].content||s.push({tagName:n(r.content[0].content[1]),openedBraces:0}):0<s.length&&"punctuation"===r.type&&"{"===r.content?s[s.length-1].openedBraces++:0<s.length&&0<s[s.length-1].openedBraces&&"punctuation"===r.type&&"}"===r.content?s[s.length-1].openedBraces--:g=!0),(g||"string"==typeof r)&&0<s.length&&0===s[s.length-1].openedBraces&&(o=n(r),i<e.length-1&&("string"==typeof e[i+1]||"plain-text"===e[i+1].type)&&(o+=n(e[i+1]),e.splice(i+1,1)),0<i&&("string"==typeof e[i-1]||"plain-text"===e[i-1].type)&&(o=n(e[i-1])+o,e.splice(i-1,1),i--),e[i]=new t.Token("plain-text",o,null,o)),r.content&&"string"!=typeof r.content&&a(r.content)}};t.hooks.add("after-tokenize",(function(t){"jsx"!==t.language&&"tsx"!==t.language||a(t.tokens)}))}(t)}(t.exports=a).displayName="jsx",a.aliases=[]},406:function(t,e,n){"use strict";function a(t){t.languages.typescript=t.languages.extend("javascript",{keyword:/\b(?:abstract|as|async|await|break|case|catch|class|const|constructor|continue|debugger|declare|default|delete|do|else|enum|export|extends|finally|for|from|function|get|if|implements|import|in|instanceof|interface|is|keyof|let|module|namespace|new|null|of|package|private|protected|public|readonly|return|require|set|static|super|switch|this|throw|try|type|typeof|var|void|while|with|yield)\b/,builtin:/\b(?:string|Function|any|number|boolean|Array|symbol|console|Promise|unknown|never)\b/}),t.languages.ts=t.languages.typescript}(t.exports=a).displayName="typescript",a.aliases=["ts"]},554:function(t,e,n){"use strict";var a=n(403),s=n(406);function i(t){t.register(a),t.register(s);var e=t.util.clone(t.languages.typescript);t.languages.tsx=t.languages.extend("jsx",e)}(t.exports=i).displayName="tsx",i.aliases=[]}}]);