(window.webpackJsonp=window.webpackJsonp||[]).push([[74],{483:function(e,n,a){"use strict";function t(e){function n(e){return RegExp("(\\()"+e+"(?=[\\s\\)])")}function a(e){return RegExp("([\\s([])"+e+"(?=[\\s)])")}var t,s,i,r,o,l,p,d,u;t=e,s="[-+*/_~!@$%^=<>{}\\w]+",i="(\\()",r="(?=\\))",o="(?=\\s)",l={heading:{pattern:/;;;.*/,alias:["comment","title"]},comment:/;.*/,string:{pattern:/"(?:[^"\\]|\\.)*"/,greedy:!0,inside:{argument:/[-A-Z]+(?=[.,\s])/,symbol:RegExp("`"+s+"'")}},"quoted-symbol":{pattern:RegExp("#?'"+s),alias:["variable","symbol"]},"lisp-property":{pattern:RegExp(":"+s),alias:"property"},splice:{pattern:RegExp(",@?"+s),alias:["symbol","variable"]},keyword:[{pattern:RegExp(i+"(?:(?:lexical-)?let\\*?|(?:cl-)?letf|if|when|while|unless|cons|cl-loop|and|or|not|cond|setq|error|message|null|require|provide|use-package)"+o),lookbehind:!0},{pattern:RegExp(i+"(?:for|do|collect|return|finally|append|concat|in|by)"+o),lookbehind:!0}],declare:{pattern:n("declare"),lookbehind:!0,alias:"keyword"},interactive:{pattern:n("interactive"),lookbehind:!0,alias:"keyword"},boolean:{pattern:a("(?:t|nil)"),lookbehind:!0},number:{pattern:a("[-+]?\\d+(?:\\.\\d*)?"),lookbehind:!0},defvar:{pattern:RegExp(i+"def(?:var|const|custom|group)\\s+"+s),lookbehind:!0,inside:{keyword:/^def[a-z]+/,variable:RegExp(s)}},defun:{pattern:RegExp(i+"(?:cl-)?(?:defun\\*?|defmacro)\\s+"+s+"\\s+\\([\\s\\S]*?\\)"),lookbehind:!0,inside:{keyword:/^(?:cl-)?def\S+/,arguments:null,function:{pattern:RegExp("(^\\s)"+s),lookbehind:!0},punctuation:/[()]/}},lambda:{pattern:RegExp(i+"lambda\\s+\\((?:&?"+s+"\\s*)*\\)"),lookbehind:!0,inside:{keyword:/^lambda/,arguments:null,punctuation:/[()]/}},car:{pattern:RegExp(i+s),lookbehind:!0},punctuation:[/(['`,]?\(|[)\[\]])/,{pattern:/(\s)\.(?=\s)/,lookbehind:!0}]},p={"lisp-marker":RegExp("&[-+*/_~!@$%^=<>{}\\w]+"),rest:{argument:{pattern:RegExp(s),alias:"variable"},varform:{pattern:RegExp(i+s+"\\s+\\S[\\s\\S]*"+r),lookbehind:!0,inside:{string:l.string,boolean:l.boolean,number:l.number,symbol:l.symbol,punctuation:/[()]/}}}},d="\\S+(?:\\s+\\S+)*",u={pattern:RegExp(i+"[\\s\\S]*"+r),lookbehind:!0,inside:{"rest-vars":{pattern:RegExp("&(?:rest|body)\\s+"+d),inside:p},"other-marker-vars":{pattern:RegExp("&(?:optional|aux)\\s+"+d),inside:p},keys:{pattern:RegExp("&key\\s+"+d+"(?:\\s+&allow-other-keys)?"),inside:p},argument:{pattern:RegExp(s),alias:"variable"},punctuation:/[()]/}},l.lambda.inside.arguments=u,l.defun.inside.arguments=t.util.clone(u),l.defun.inside.arguments.inside.sublist=u,t.languages.lisp=l,t.languages.elisp=l,t.languages.emacs=l,t.languages["emacs-lisp"]=l}(e.exports=t).displayName="lisp",t.aliases=[]}}]);