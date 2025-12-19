// 导入静态可视化模块和全局变量
const VIS = require("./staticVisualization");

/**
 * 主函数：程序入口，负责协调整个可视化流程
 * 执行顺序：加载数据 → 渲染静态可视化 → 渲染主可视化层 → 挂载交互功能
 */
async function main() {
  // 等待数据加载完成（异步操作）
  await VIS.loadData();
  // 渲染静态可视化内容
  VIS.renderStaticVisualization();
  // 渲染主可视化图层并获取图层实例
  const mainLayer = renderMainVisualization();
  // 为主图层挂载交互功能
  mountInteraction(mainLayer);
}

/**
 * 渲染主可视化图层
 * @returns {Libra.Layer} 返回创建好的主图层实例
 */
function renderMainVisualization() {
  // 选择页面上 ID 为 LibraPlayground 下的 SVG 元素
  const svg = d3.select("#LibraPlayground svg");

  // 创建主可视化图层
  const mainLayer = Libra.Layer.initialize("D3Layer", {
    name: "mainLayer",          // 图层名称
    width: globalThis.WIDTH,    // 图层宽度（使用全局变量）
    height: globalThis.HEIGHT,  // 图层高度（使用全局变量）
    offset: {                   // 图层偏移量（边距）
      x: globalThis.MARGIN.left,
      y: globalThis.MARGIN.top
    },
    container: svg.node(),      // 图层挂载的容器（SVG DOM 节点）
  });
  
  // 获取图层的图形容器并转换为 D3 选择集
  const g = d3.select(mainLayer.getGraphic());

  // 绘制散点图（核心可视化逻辑）
  g.selectAll("circle")
    .data(globalThis.data)      // 绑定全局数据
    .join("circle")             // 数据绑定：创建/更新/删除 circle 元素
    .attr("class", "mark")      // 设置元素类名
    .attr("cx", (d) => globalThis.x(d[globalThis.FIELD_X]))  // 设置圆心 X 坐标（使用全局比例尺）
    .attr("cy", (d) => globalThis.y(d[globalThis.FIELD_Y]))  // 设置圆心 Y 坐标（使用全局比例尺）
    .attr("fill", (d) => globalThis.color(d[globalThis.FIELD_COLOR]))  // 设置填充颜色（使用全局颜色比例尺）
    .attr("fill-opacity", 0.7)  // 设置填充透明度
    .attr("r", 3);              // 设置圆的半径

  // 返回创建好的主图层实例
  return mainLayer;
}

/**
 * 为可视化图层挂载交互功能
 * @param {Libra.Layer} layer - 需要挂载交互的图层实例
 */
async function mountInteraction(layer) {
  // 构建并挂载悬停交互工具（HoverInstrument）
  Libra.Interaction.build({
    inherit: "HoverInstrument", // 继承的交互工具类型（悬停工具）
    layers: [layer],            // 应用该交互的图层列表
    sharedVar: {                // 交互工具的共享变量配置
      tooltip: {                // 提示框（Tooltip）配置
        image: (d) => d.image,  // 提示框中显示的图片（从数据中获取）
        offset: {               // 提示框偏移量（调整提示框显示位置）
          x: -70 - globalThis.MARGIN.left,
          y: -100 - globalThis.MARGIN.top,
        },
      },
    },
  });
  
  // 创建交互历史追踪（异步操作，记录用户交互行为）
  await Libra.createHistoryTrrack();
}

// 执行主函数，启动整个可视化程序
main();
