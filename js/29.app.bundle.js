(window.webpackJsonp=window.webpackJsonp||[]).push([[29],{647:function(e,n,s){"use strict";function t(e){e.languages.css.selector={pattern:e.languages.css.selector,inside:{"pseudo-element":/:(?:after|before|first-letter|first-line|selection)|::[-\w]+/,"pseudo-class":/:[-\w]+/,class:/\.[-:.\w]+/,id:/#[-:.\w]+/,attribute:{pattern:/\[(?:[^[\]"']|("|')(?:\\(?:\r\n|[\s\S])|(?!\1)[^\\\r\n])*\1)*\]/,greedy:!0,inside:{punctuation:/^\[|\]$/,"case-sensitivity":{pattern:/(\s)[si]$/i,lookbehind:!0,alias:"keyword"},namespace:{pattern:/^(\s*)[-*\w\xA0-\uFFFF]*\|(?!=)/,lookbehind:!0,inside:{punctuation:/\|$/}},attribute:{pattern:/^(\s*)[-\w\xA0-\uFFFF]+/,lookbehind:!0},value:[/("|')(?:\\(?:\r\n|[\s\S])|(?!\1)[^\\\r\n])*\1/,{pattern:/(=\s*)[-\w\xA0-\uFFFF]+(?=\s*$)/,lookbehind:!0}],operator:/[|~*^$]?=/}},"n-th":[{pattern:/(\(\s*)[+-]?\d*[\dn](?:\s*[+-]\s*\d+)?(?=\s*\))/,lookbehind:!0,inside:{number:/[\dn]+/,operator:/[+-]/}},{pattern:/(\(\s*)(?:even|odd)(?=\s*\))/i,lookbehind:!0}],punctuation:/[()]/}},e.languages.insertBefore("css","property",{variable:{pattern:/(^|[^-\w\xA0-\uFFFF])--[-_a-z\xA0-\uFFFF][-\w\xA0-\uFFFF]*/i,lookbehind:!0}}),e.languages.insertBefore("css","function",{operator:{pattern:/(\s)[+\-*\/](?=\s)/,lookbehind:!0},hexcode:/#[\da-f]{3,8}/i,entity:/\\[\da-f]{1,8}/i,unit:{pattern:/(\d)(?:%|[a-z]+)/,lookbehind:!0},number:/-?[\d.]+/})}e.exports=t,t.displayName="cssExtras",t.aliases=[]}}]);