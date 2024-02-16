const content = document.querySelector('#content')
const isPrompt = row => {
  const p = row.filter(x => x).length === 1
  return p
}
const agg_span_attns = (attns) => {
    const sel = window.getSelection()
    const fromNode = sel.anchorNode.parentNode
    const toNode = sel.extentNode.parentNode
    const fromIdx = Array.from(content.children).indexOf(fromNode)
    const toIdx = Array.from(content.children).indexOf(toNode)
    const range = [fromIdx, toIdx]
    range.sort((a, b) => a - b)
    const vec = tokens.map((x, i) => (i >= range[0] && i <= range[1]) ? 1 : 0)
    return  math.multiply(vec, attns)
}
const updateSidebar = (attn_vec) => {
  const attnTokenPairs = tokens.map((token, i) => ({token, attn: math.abs(attn_vec[i])}))
  const sortedPairs = attnTokenPairs.sort((a, b) => b.attn - a.attn)
  const maxAttention = sortedPairs[0].attn
  const sortedTokens = sortedPairs.slice(0, 20)
  const sidebar = document.getElementById('sidebar');
  sidebar.innerHTML = `<div class="bar-chart">` +
  sortedTokens.map(tokenData => {
      const barWidth = (tokenData.attn / maxAttention) * 100;
      let tokenTxt = tokenData.token.replace("\n", "[NEWLINE]")
      tokenTxt = tokenTxt.replace("<", "[")
      tokenTxt = tokenTxt.replace(">", "]")
      return `
          <div class="bar-container">
              <span class="bar-token">${tokenTxt}</span>
              <div class="bar" style="width: ${barWidth}%;">
                  <span class="bar-score">${tokenData.attn.toFixed(2)}</span>
              </div>
          </div>`;
  }).join('') +
  `</div>`;
}
const fromSparse = (size, indices, values) => {
  let xs = Array.from({length: size}, () => Array.from({length: size}, () => 0))
  indices.forEach(([i, j], x) => {
    xs[i][j] = values[x]
  })
  return xs
}
let tokens = []
let attn_m = []
const redraw = () => {
  if (!window.getSelection().isCollapsed) {
    let attn_vec = agg_span_attns(attn_m)
    Array.from(content.children).forEach((node, i) => {
      const attn = attn_vec[i]
      node.style.setProperty('--attention', Math.min(1, attn * 5).toFixed(2))
    })
    updateSidebar(attn_vec)
  } else {
    Array.from(content.children).forEach((node, i) => {
      node.style.setProperty('--attention', '0')
    })
  }
}
document.addEventListener('mousemove', redraw)
document.addEventListener('mouseup', redraw)

let currentIndex = 0; // Global variable to track the current index

// Function to fetch and update data
function fetchData(index) {
  fetch(`/attention/${index}`).then(async res => {
    const data = await res.json();
    console.log(data);
    tokens = data.tokens;
    attn_m = fromSparse(tokens.length, data.attn_indices, data.attn_values);
    const content = document.getElementById('content');
    content.innerHTML = ''; // Clear existing content
    data.tokens.forEach((t, i) => {
      const token = document.createElement('span');
      let tokenTxt = t.replace("\n", "[NEWLINE]\n").replace("<", "[").replace(">", "]");
      token.innerText = tokenTxt;
      token.classList.add('prompt');
      content.appendChild(token);
    });
  });
}

// Initial fetch
fetchData(currentIndex);

// Event listener for the button
document.getElementById('nextButton').addEventListener('click', () => {
  currentIndex++; // Increment the index
  fetchData(currentIndex); // Fetch and update data for the new index
});
