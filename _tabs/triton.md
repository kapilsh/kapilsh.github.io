---
# the default layout is 'page'
title: Triton IR Parser
icon: fas fa-solid fa-microchip
order: 3
---

Triton IR Parser will parse the MLIR generated by triton compiler to display sections with instructions/operations and their corresponding file information.

<div>
    <div style="font-size: 8pt; font-family: monospace">
        <textarea id="input-box"
            style="width: 100%; height: 300px; background-color: #151515"></textarea>
    </div>
    <button id="parse-button" style="background-color: #151515">Parse</button>
    <div class="language-text highlighter-rouge">
        <div class="code-header">
            <span data-label-text="Parsed Output">
                <i class="fas fa-code fa-fw small"></i>
            </span>
            <button aria-label="copy" data-title-succeed="Copied!">
                <i class="far fa-clipboard">
                </i>
            </button>
        </div>
        <div id="output-div" style="white-space: pre; font-family: monospace" class="highlight">
        </div>
    </div>
<script>
(function () {
function parseMlir(t){let e=t.split("\n"),o={};for(let l of e){let r=l.match(/#loc(\d+) = loc\((.*)\)/);if(r){let n=r[1],c=r[2].replace(/"/g,"");o[n]={fileInfo:c,code:[]}}}for(let i of e){let d=i.match(/.*loc\(#loc(\d+)\)/);if(d){let u=d[1];u in o&&o[u].code.push(i.trim())}}let p="",a=1;for(let s in o){let f=o[s];for(let g of(p+=`<tr><td class="rouge-gutter gl"><pre class="lineno">${s}</pre></td><td class="rouge-code"><pre>Section ${a}: ${f.fileInfo}
`,f.code))p+=`  ${g}
`;p+="</pre></td></tr>",a++}return p}const inputBox=document.getElementById("input-box"),outputDiv=document.getElementById("output-div"),parseButton=document.getElementById("parse-button");parseButton.addEventListener("click",()=>{let t=inputBox.value,e=parseMlir(t);outputDiv.innerHTML=`<code><table class="rouge-table"><tbody>${e}</tbody></table></code>`});
})();
</script>
</div>
