// 模拟Libra的Layer类
export class Layer {
  static initialize(type, options) {
    return new Layer(options);
  }
  constructor(options) {
    this.name = options.name;
    this.container = options.container;
    this.graphic = options.graphic;
  }
  getGraphic() {
    return this.graphic;
  }
}

// 模拟Libra的Interaction类
export class Interaction {
  static build(config) {
    return new Interaction(config);
  }
  constructor(config) {
    this.config = config;
    this.layers = config.layers;
    this.sharedVar = config.sharedVar || {};
  }
  attachToView(view) {
    this.bindEvents(view);
  }
  bindEvents(view) {
    // 绑定Hover事件
    if (this.config.inherit === 'HoverInstrument') {
      const layerGraphic = this.layers[0].getGraphic();
      const elements = layerGraphic.querySelectorAll('circle');
      elements.forEach(el => {
        el.addEventListener('mouseover', (e) => {
          const data = e.target.__data__;
          el.setAttribute('r', this.sharedVar.highlightStyle.r || 5);
          el.setAttribute('stroke', this.sharedVar.highlightStyle.stroke || 'white');
          el.setAttribute('stroke-width', this.sharedVar.highlightStyle.strokeWidth || 1);
          // 显示tooltip
          let tooltip = document.getElementById('libra-tooltip');
          if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'libra-tooltip';
            tooltip.style.position = 'absolute';
            tooltip.style.background = 'white';
            tooltip.style.padding = '5px';
            tooltip.style.border = '1px solid #ccc';
            tooltip.style.borderRadius = '3px';
            tooltip.style.pointerEvents = 'none';
            document.body.appendChild(tooltip);
          }
          tooltip.innerHTML = this.sharedVar.tooltip(data);
          tooltip.style.left = (e.pageX + 10) + 'px';
          tooltip.style.top = (e.pageY + 10) + 'px';
        });
        el.addEventListener('mouseout', (e) => {
          const data = e.target.__data__;
          e.target.setAttribute('r', 3);
          e.target.setAttribute('stroke', 'none');
          const tooltip = document.getElementById('libra-tooltip');
          if (tooltip) tooltip.innerHTML = '';
        });
      });
    }
    // 绑定选择事件（简化版）
    if (this.config.inherit === 'ClickInstrument') {
      const layerGraphic = this.layers[0].getGraphic();
      const elements = layerGraphic.querySelectorAll('circle');
      elements.forEach(el => {
        el.addEventListener('click', (e) => {
          const data = e.target.__data__;
          el.setAttribute('fill', this.sharedVar.selectStyle.fill || 'orange');
          el.setAttribute('stroke', this.sharedVar.selectStyle.stroke || 'red');
          el.setAttribute('stroke-width', this.sharedVar.selectStyle.strokeWidth || 2);
        });
      });
    }
  }
}

// 模拟Libra的Service类（历史管理简化版）
export class Service {
  static initialize(type, options) {
    return new Service(options);
  }
  constructor(options) {
    this.maxHistory = options.maxHistory || 20;
    this.history = [];
    this.index = -1;
  }
  undo() {
    alert('撤销操作（简化版）');
  }
  redo() {
    alert('重做操作（简化版）');
  }
}