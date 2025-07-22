# transportation_optimizer.py
"""
物流运输优化器 - 基于线性规划的成本最小化工具
GitHub: https://github.com/[你的用户名]/Logistics-Optimizer
"""
import pulp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from io import StringIO

# 设置页面标题
st.set_page_config(page_title="🚚 智能物流优化系统", layout="wide")
st.title("🚚 物流运输成本优化器")
st.markdown("""
**基于线性规划(Linear Programming)的运输问题求解工具**  
使用PuLP库实现最小化运输成本 - 工程管理运筹学应用
""")

# 在侧边栏添加说明
with st.sidebar:
    st.header("使用说明")
    st.markdown("""
    1. **上传数据**：提供CSV格式的运费表和供需数据
    2. **设置参数**：调整求解器参数
    3. **运行优化**：点击求解按钮获取最优方案
    4. **查看结果**：分析优化方案和成本节约
    """)
    st.divider()
    st.caption("技术栈: Python, PuLP, Pandas, Streamlit")
    st.caption("运筹学应用: 线性规划 | 运输问题 | 成本优化")

# 示例数据
def create_example_data():
    # 运费数据
    cost_data = {
        '仓库': ['长沙', '长沙', '长沙', '株洲', '株洲', '株洲', '湘潭', '湘潭', '湘潭'],
        '城市': ['武汉', '南昌', '广州', '武汉', '南昌', '广州', '武汉', '南昌', '广州'],
        '运费(元/吨)': [150, 180, 280, 120, 140, 240, 200, 160, 220]
    }
    cost_df = pd.DataFrame(cost_data)
    
    # 供应数据
    supply_data = {
        '仓库': ['长沙', '株洲', '湘潭'],
        '供应量(吨)': [300, 400, 200]
    }
    supply_df = pd.DataFrame(supply_data)
    
    # 需求数据
    demand_data = {
        '城市': ['武汉', '南昌', '广州'],
        '需求量(吨)': [350, 250, 300]
    }
    demand_df = pd.DataFrame(demand_data)
    
    return cost_df, supply_df, demand_df

# 创建优化模型
def create_optimization_model(cost_df, supply_df, demand_df):
    # 创建问题实例
    prob = pulp.LpProblem("物流运输成本优化", pulp.LpMinimize)
    
    # 获取仓库和城市列表
    warehouses = supply_df['仓库'].tolist()
    cities = demand_df['城市'].tolist()
    
    # 创建运输量变量
    routes = [(w, c) for w in warehouses for c in cities]
    x = pulp.LpVariable.dicts("运输量", routes, lowBound=0, cat='Continuous')
    
    # 创建目标函数：最小化总运费
    prob += pulp.lpSum([x[w, c] * cost_df.loc[
        (cost_df['仓库'] == w) & (cost_df['城市'] == c), '运费(元/吨)'].values[0] 
        for (w, c) in routes])
    
    # 添加供应约束：每个仓库的运出量不超过供应量
    for w in warehouses:
        prob += pulp.lpSum([x[w, c] for c in cities]) <= supply_df.loc[
            supply_df['仓库'] == w, '供应量(吨)'].values[0]
    
    # 添加需求约束：每个城市的需求必须满足
    for c in cities:
        prob += pulp.lpSum([x[w, c] for w in warehouses]) >= demand_df.loc[
            demand_df['城市'] == c, '需求量(吨)'].values[0]
    
    return prob, x

# 解决优化问题
def solve_optimization(prob):
    # 使用CBC求解器
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    return pulp.LpStatus[prob.status]

# 处理结果
def process_results(prob, x, cost_df, supply_df, demand_df):
    # 获取仓库和城市列表
    warehouses = supply_df['仓库'].tolist()
    cities = demand_df['城市'].tolist()
    
    # 提取解决方案
    solution = []
    total_cost = 0
    
    for w in warehouses:
        for c in cities:
            var_value = pulp.value(x[w, c])
            if var_value > 0:
                cost = cost_df.loc[
                    (cost_df['仓库'] == w) & (cost_df['城市'] == c), '运费(元/吨)'].values[0]
                solution.append({
                    '仓库': w,
                    '城市': c,
                    '运输量(吨)': var_value,
                    '运费(元/吨)': cost,
                    '总运费(元)': var_value * cost
                })
                total_cost += var_value * cost
    
    solution_df = pd.DataFrame(solution)
    
    # 计算成本节约（与原始方案相比）
    original_cost = 0
    for _, row in cost_df.iterrows():
        w = row['仓库']
        c = row['城市']
        # 假设原始方案是平均分配
        min_supply = min(supply_df[supply_df['仓库'] == w]['供应量(吨)'].values[0],
                         demand_df[demand_df['城市'] == c]['需求量(吨)'].values[0])
        original_cost += min_supply * row['运费(元/吨)'] / len(warehouses)
    
    cost_saving = original_cost - total_cost
    saving_percent = (cost_saving / original_cost) * 100 if original_cost > 0 else 0
    
    return solution_df, total_cost, cost_saving, saving_percent

# 可视化结果
def visualize_results(solution_df, total_cost, cost_saving, saving_percent):
    # 创建布局
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("优化结果概览")
        st.metric("总运输成本", f"¥{total_cost:,.0f}元")
        st.metric("成本节约", f"¥{cost_saving:,.0f}元", f"{saving_percent:.1f}%")
        
        # 显示优化方案
        st.subheader("最优运输方案")
        st.dataframe(solution_df.style.format({
            '运输量(吨)': '{:.0f}',
            '运费(元/吨)': '{:.0f}',
            '总运费(元)': '{:,.0f}'
        }), height=300)
    
    with col2:
        st.subheader("运输量分布")
        
        # 创建矩阵图
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 创建热力图数据
        warehouses = solution_df['仓库'].unique()
        cities = solution_df['城市'].unique()
        data = pd.DataFrame(0, index=warehouses, columns=cities)
        
        for _, row in solution_df.iterrows():
            data.at[row['仓库'], row['城市']] = row['运输量(吨)']
        
        # 绘制热力图
        im = ax.imshow(data.values, cmap="YlGn")
        
        # 设置标签
        ax.set_xticks(np.arange(len(cities)), labels=cities)
        ax.set_yticks(np.arange(len(warehouses)), labels=warehouses)
        ax.set_xlabel("城市")
        ax.set_ylabel("仓库")
        
        # 添加数值标签
        for i in range(len(warehouses)):
            for j in range(len(cities)):
                text = ax.text(j, i, f"{data.iloc[i, j]:.0f}",
                              ha="center", va="center", color="black")
        
        plt.title("仓库-城市运输量分配 (吨)")
        st.pyplot(fig)
        
        # 成本构成饼图
        if not solution_df.empty:
            st.subheader("成本构成分析")
            cost_dist = solution_df.groupby('仓库')['总运费(元)'].sum()
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.pie(cost_dist, labels=cost_dist.index, autopct='%1.1f%%')
            ax2.set_title("各仓库运输成本占比")
            st.pyplot(fig2)

# 主应用逻辑
def main_app():
    # 数据上传部分
    st.header("1. 输入数据")
    
    # 创建选项卡
    tab1, tab2, tab3 = st.tabs(["上传CSV", "手动输入", "使用示例数据"])
    
    with tab1:
        st.subheader("上传CSV文件")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            cost_file = st.file_uploader("运费表 (仓库, 城市, 运费)", type=["csv"])
        with col2:
            supply_file = st.file_uploader("仓库供应量表", type=["csv"])
        with col3:
            demand_file = st.file_uploader("城市需求量表", type=["csv"])
        
        if cost_file and supply_file and demand_file:
            cost_df = pd.read_csv(cost_file)
            supply_df = pd.read_csv(supply_file)
            demand_df = pd.read_csv(demand_file)
        else:
            cost_df, supply_df, demand_df = None, None, None
    
    with tab2:
        st.subheader("手动输入数据")
        
        with st.expander("运费表"):
            cost_data = st.data_editor(
                pd.DataFrame([{"仓库": "", "城市": "", "运费(元/吨)": 0}]),
                num_rows="dynamic",
                column_config={
                    "仓库": st.column_config.TextColumn(width="medium"),
                    "城市": st.column_config.TextColumn(width="medium"),
                    "运费(元/吨)": st.column_config.NumberColumn(width="small")
                },
                key="cost_editor"
            )
            cost_df = cost_data.dropna(how='all') if not cost_data.empty else None
        
        with st.expander("仓库供应量"):
            supply_data = st.data_editor(
                pd.DataFrame([{"仓库": "", "供应量(吨)": 0}]),
                num_rows="dynamic",
                column_config={
                    "仓库": st.column_config.TextColumn(width="medium"),
                    "供应量(吨)": st.column_config.NumberColumn(width="small")
                },
                key="supply_editor"
            )
            supply_df = supply_data.dropna(how='all') if not supply_data.empty else None
        
        with st.expander("城市需求量"):
            demand_data = st.data_editor(
                pd.DataFrame([{"城市": "", "需求量(吨)": 0}]),
                num_rows="dynamic",
                column_config={
                    "城市": st.column_config.TextColumn(width="medium"),
                    "需求量(吨)": st.column_config.NumberColumn(width="small")
                },
                key="demand_editor"
            )
            demand_df = demand_data.dropna(how='all') if not demand_data.empty else None
    
    with tab3:
        st.subheader("使用示例数据")
        if st.button("加载示例数据"):
            cost_df, supply_df, demand_df = create_example_data()
            st.session_state.example_loaded = True
        
        if st.session_state.get('example_loaded', False):
            st.info("示例数据已加载！")
            cost_df, supply_df, demand_df = create_example_data()
        else:
            cost_df, supply_df, demand_df = None, None, None
    
    # 显示数据预览
    if cost_df is not None and supply_df is not None and demand_df is not None:
        st.subheader("数据预览")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("运费表")
            st.dataframe(cost_df, height=200)
        with col2:
            st.write("仓库供应量")
            st.dataframe(supply_df, height=200)
        with col3:
            st.write("城市需求量")
            st.dataframe(demand_df, height=200)
        
        # 求解按钮
        st.divider()
        st.header("2. 运行优化")
        
        if st.button("求解运输优化问题", type="primary"):
            with st.spinner("正在优化运输方案..."):
                # 创建并求解模型
                prob, x = create_optimization_model(cost_df, supply_df, demand_df)
                status = solve_optimization(prob)
                
                if status == "Optimal":
                    st.success("找到最优解！")
                    solution_df, total_cost, cost_saving, saving_percent = process_results(
                        prob, x, cost_df, supply_df, demand_df)
                    
                    # 显示结果
                    st.divider()
                    st.header("3. 优化结果")
                    visualize_results(solution_df, total_cost, cost_saving, saving_percent)
                    
                    # 添加下载按钮
                    csv = solution_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="下载优化方案 (CSV)",
                        data=csv,
                        file_name='optimal_transport_plan.csv',
                        mime='text/csv'
                    )
                else:
                    st.error(f"求解失败: {status}")
                    st.write("可能原因: 供需不平衡或数据错误")
    else:
        st.info("请提供完整的数据以开始优化")

# 初始化会话状态
if 'example_loaded' not in st.session_state:
    st.session_state.example_loaded = False

# 运行应用
if __name__ == "__main__":
    main_app()
