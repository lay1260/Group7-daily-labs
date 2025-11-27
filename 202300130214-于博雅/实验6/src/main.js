// 导入模拟的Libra模块和D3
import { Layer, Interaction, Service } from './lib/libra.js';
import * as d3 from 'd3';

// 1. 创建SVG容器
const svg = d3.select('body')
  .append('svg')
  .attr('width', 800)
  .attr('height', 600)
  .style('border', '1px solid #eee'); // 加边框方便看

// 2. 初始化Libra的Layer
const mainLayer = Layer.initialize('D3Layer', {
  name: 'mnistLayer',
  container: svg.node(),
  graphic: svg.append('g').attr('class', 'points').node() // 散点的父容器
});

// 3. 生成模拟的MNIST数据（不用下载真实数据，直接模拟）
const mockMNISTData = Array.from({ length: 200 }, () => {
  const digit = Math.floor(Math.random() * 10); // 随机数字0-9
  return {
    x: (Math.random() - 0.5) * 70, // 坐标范围-35到35
    y: (Math.random() - 0.5) * 50,
    digit: digit,
    image_url: `https://picsum.photos/50/50?random=${digit}` // 模拟图片链接
  };
});

// 4. 渲染散点图（基于Libra的Layer）
const g = d3.select(mainLayer.getGraphic());
g.selectAll('circle')
  .data(mockMNISTData)
  .join('circle')
  .attr('cx', d => d.x * 10 + 400) // 缩放坐标到SVG中心
  .attr('cy', d => d.y * 10 + 300)
  .attr('r', 3)
  .attr('fill', d => d3.schemeCategory10[d.digit]) // 按数字配色
  .each(function(d) {
    this.__data__ = d; // 绑定数据到DOM元素（方便交互获取）
  });

// 5. 实现Hover交互（悬停放大+显示tooltip）
const hoverInst = Interaction.build({
  inherit: 'HoverInstrument',
  layers: [mainLayer],
  sharedVar: {
    tooltip: d => `
      <div style="text-align:center">
        <img src="${d.image_url}" width="50" height="50"/><br>
        数字：${d.digit}
      </div>
    `,
    highlightStyle: { r: 5, stroke: 'white', strokeWidth: 1 }
  }
});
hoverInst.attachToView(svg.node());

// 6. 实现点击选择交互
const selectInst = Interaction.build({
  inherit: 'ClickInstrument',
  layers: [mainLayer],
  sharedVar: {
    selectStyle: { fill: 'orange', stroke: 'red', strokeWidth: 2 }
  }
});
selectInst.attachToView(svg.node());

// 7. 初始化历史管理（简化版）
const historyService = Service.initialize('HistoryManager', { maxHistory: 20 });
// 绑定撤销/重做快捷键
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === 'z') {
    historyService.undo();
  } else if (e.ctrlKey && e.key === 'y') {
    historyService.redo();
  }
});