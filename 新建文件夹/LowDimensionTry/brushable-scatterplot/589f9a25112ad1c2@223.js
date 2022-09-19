// https://observablehq.com/@d3/brushable-scatterplot@223
function _1(data,md){return(
md`# Brushable Scatterplot

This chart shows the inverse relationship between engine power (*y*-axis) and fuel efficiency (*x*-axis) in ${data.length} cars from 1970–1982. Brushing this scatterplot will show the selected data points. 
Thanks to [John Alexis Guerra Gómez](/@john-guerra) for suggestions.`
)}

function _selection(d3,width,height,xAxis,yAxis,data,x,y)
{
  const svg = d3.create("svg")
      .attr("viewBox", [0, 0, width, height])
      .property("value", []);

  const brush = d3.brush()
      .on("start brush end", brushed);

  svg.append("g")
      .call(xAxis);

  svg.append("g")
      .call(yAxis);

  const dot = svg.append("g")
      .attr("fill", "none")
      .attr("stroke", "steelblue")
      .attr("stroke-width", 1.5)
    .selectAll("circle")
    .data(data)
    .join("circle")
      .attr("transform", d => `translate(${x(d.x)},${y(d.y)})`)
      .attr("r", 3);

  svg.call(brush);

  function brushed({selection}) {
    let value = [];
    if (selection) {
      const [[x0, y0], [x1, y1]] = selection;
      value = dot
        .style("stroke", "gray")
        .filter(d => x0 <= x(d.x) && x(d.x) < x1 && y0 <= y(d.y) && y(d.y) < y1)
        .style("stroke", "steelblue")
        .data();
    } else {
      dot.style("stroke", "steelblue");
    }
    svg.property("value", value).dispatch("input");
  }

  return svg.node();
}


function _3(selection){return(
selection
)}

function _height(){return(
600
)}

function _margin(){return(
{top: 20, right: 30, bottom: 30, left: 40}
)}

function _x(d3,data,margin,width){return(
d3.scaleLinear()
    .domain(d3.extent(data, d => d.x)).nice()
    .range([margin.left, width - margin.right])
)}

function _y(d3,data,height,margin){return(
d3.scaleLinear()
   .domain(d3.extent(data, d => d.y)).nice()
   .range([height - margin.bottom, margin.top])
)}

function _xAxis(height,margin,d3,x,width,data){return(
g => g
    .attr("transform", `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x))
    .call(g => g.select(".domain").remove())
    .call(g => g.append("text")
        .attr("x", width - margin.right)
        .attr("y", -4)
        .attr("fill", "#000")
        .attr("font-weight", "bold")
        .attr("text-anchor", "end")
        .text(data.x))
)}

function _yAxis(margin,d3,y,data){return(
g => g
    .attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y))
    .call(g => g.select(".domain").remove())
    .call(g => g.select(".tick:last-of-type text").clone()
        .attr("x", 4)
        .attr("text-anchor", "start")
        .attr("font-weight", "bold")
        .text(data.y))
)}

async function _data(d3,FileAttachment){return(
Object.assign(d3.csvParse(await FileAttachment("cars-2.csv").text(), ({Name: name, Miles_per_Gallon: x, Horsepower: y}) => ({name, x: +x, y: +y})), {x: "Miles per Gallon", y: "Horsepower"})
)}

export default function define(runtime, observer) {
  const main = runtime.module();
  function toString() { return this.url; }
  const fileAttachments = new Map([
    ["cars-2.csv", {url: new URL("./files/53c407ee531bab128477148c9e28c49dd06bf83a93ae317e58dbb9fc819db0d4f6c4fb9646fa2fe20faad76addee20cfc360eab2362eeaec3340a5e4655b9996", import.meta.url), mimeType: null, toString}]
  ]);
  main.builtin("FileAttachment", runtime.fileAttachments(name => fileAttachments.get(name)));
  main.variable(observer()).define(["data","md"], _1);
  main.variable(observer("viewof selection")).define("viewof selection", ["d3","width","height","xAxis","yAxis","data","x","y"], _selection);
  main.variable(observer("selection")).define("selection", ["Generators", "viewof selection"], (G, _) => G.input(_));
  main.variable(observer()).define(["selection"], _3);
  main.variable(observer("height")).define("height", _height);
  main.variable(observer("margin")).define("margin", _margin);
  main.variable(observer("x")).define("x", ["d3","data","margin","width"], _x);
  main.variable(observer("y")).define("y", ["d3","data","height","margin"], _y);
  main.variable(observer("xAxis")).define("xAxis", ["height","margin","d3","x","width","data"], _xAxis);
  main.variable(observer("yAxis")).define("yAxis", ["margin","d3","y","data"], _yAxis);
  main.variable(observer("data")).define("data", ["d3","FileAttachment"], _data);
  return main;
}
