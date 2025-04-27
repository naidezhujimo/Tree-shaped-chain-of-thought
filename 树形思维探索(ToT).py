from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter
from sympy import symbols, Eq, solve, sympify


tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    pad_token="<|endoftext|>"  # 显式设置pad token
)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device_map="auto"
)

# 确保pad_token有效，若不存在则使用eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

import math
from graphviz import Digraph

class ToTNode:
    def __init__(self, state, parent=None, depth=0):
        self.state = state      # 当前状态字符串
        self.parent = parent    # 父节点引用
        self.children = []      # 子节点列表
        self.value = 0.0        # 累计评估值
        self.visits = 0        # 访问次数
        self.depth = depth     # 当前深度
        self.is_terminal = False  # 是否为终止节点
        
    def ucb_score(self, exploration=1.414):
        """计算UCB选择分数"""
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
    
    def to_dict(self):
        """序列化节点信息"""
        return {
            "state": self.state[:50]+"..." if len(self.state)>50 else self.state,
            "depth": self.depth,
            "value": round(self.value,2),
            "visits": self.visits,
            "is_terminal": self.is_terminal
        }

# 多路径生成
def generate_tot_prompt(current_state, method, depth):
    """生成树形推理提示模板"""
    method_templates = {
        "algebra": [
            "当前深度{depth}：请从代数消元法角度思考",
            "可能的操作方向：",
            "1. 观察方程结构，识别可消元变量",
            "2. 执行加减操作消除一个变量",
            "3. 解出剩余变量后回代",
            "请选择最合理的下一步操作并执行计算"
        ],
        "matrix": [
            "当前深度{depth}：请从矩阵运算角度思考", 
            "可能的操作方向：",
            "1. 将方程组转换为矩阵形式AX=B",
            "2. 计算系数矩阵的行列式",
            "3. 若可逆则计算A的逆矩阵",
            "请选择最合理的下一步操作并执行计算"
        ],
        "default": [
            "当前深度{depth}：请综合思考解题策略",
            "可能的操作方向：",
            "1. 分析方程间的线性关系",
            "2. 选择最优消元路径",
            "3. 验证中间步骤的正确性",
            "请选择最合理的下一步操作并执行计算"
        ]
    }
    template = "\n".join([
        "请严格遵循以下要求：",
        "1. 必须使用具体数值，禁止使用'值'等占位符",
        "2. 小数保留两位，例如：x=3.50,y=2.00",
        "3. 示例格式：",
        "<solution>",
        "1. 方程2 => x = 2y - 6",
        "2. 代入方程1得 4y - 12 + 3y = 16",
        "3. 解得 y = 4.00 => x = 2.00",
        "</solution>",
        "<final_answer>x=2.00,y=4.00</final_answer>"
    ] + method_templates.get(method, method_templates["default"]))
    
    return f"{template.format(depth=depth)}\n\n当前状态：{current_state}"

def is_terminal_state(text):
    """更宽松的终止条件判断"""
    return re.search(r"<final_answer>.*?</final_answer>", text, re.IGNORECASE|re.DOTALL)

def expand_node(node, beam_width=2):
    """扩展单个节点"""
    candidates = []
    for method in ["algebra", "matrix", "default"]:
        # 生成提示
        prompt = generate_tot_prompt(
            node.state.split("<solution>")[-1],  # 仅传递推导步骤部分
            method, 
            node.depth+1
        )
                
        # 模型生成
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=400,
            temperature=max(0.5, 1.0 - 0.05*node.depth),  # 深度越大生成越稳定
            top_p=0.9,
            repetition_penalty=1.3,  # 增强重复惩罚
            num_beams=7,        # 增加束搜索宽度
            no_repeat_ngram_size=3,  # 添加N-gram重复惩罚
            early_stopping=True,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_state = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 创建子节点
        child = ToTNode(
            state=new_state,
            parent=node,
            depth=node.depth + 1
        )
        child.is_terminal = is_terminal_state(new_state) is not None
        candidates.append(child)
    
    # 评估并选择最优分支
    evaluated = [(c, evaluate_node(c)) for c in candidates]
    evaluated.sort(key=lambda x: x[1], reverse=True)
    node.children = [c for c, _ in evaluated[:beam_width]]
    
    return node

def monte_carlo_tree_search(root, iterations=100):
    """蒙特卡洛树搜索主循环"""
    for i in range(iterations):
        # 选择阶段
        node = root
        while node.children:
            node = max(node.children, key=lambda x: x.ucb_score())
        
        # 扩展阶段
        if not node.is_terminal:
            expand_node(node)
        
        # 模拟阶段（此处简化使用评估值代替随机模拟）
        reward = evaluate_node(node)
        
        # 回溯更新
        while node is not None:
            node.visits += 1
            node.value += reward / (node.depth + 1)  # 深度归一化
            node = node.parent
        # 调试输出
        print(f"Iteration {i+1}: Root visits={root.visits}, value={root.value:.2f}")
    return root

def evaluate_node(node):
    """综合评估节点质量"""
    # 数学验证得分
    math_score = 0.0
    if node.is_terminal:
        answer = extract_answer(node.state)
        try:
            is_valid = mathematical_verification(node.parent.state, answer)
            # 允许部分正确性得分
            math_score = 0.8 if is_valid else 0.2  # 即使验证失败也给基础分
        except Exception as e:
            print(f"数学验证异常: {str(e)}")
            math_score = 0.3
    
    # 语义连贯性评估
    prompt = f"""请评估以下推导的合理性（0-1分）：
{node.state}
评分："""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs.input_ids,
                             attention_mask=inputs.attention_mask,
                             max_new_tokens=3,
                             pad_token_id=tokenizer.pad_token_id)
    score_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        semantic_score = min(max(float(re.findall(r"\d\.?\d*", score_text)[-1]), 1.0), 0.0)
    except:
        semantic_score = 0.3  # 默认值
    
    # 深度衰减因子
    depth_factor = 0.95 ** node.depth

    # if node.is_terminal:
    #     print(f"\n[DEBUG] 评估终端节点:")
    #     print(f"原始内容: {node.state[:200]}...")
    #     print(f"提取结果: {answer}")
    #     print(f"数学验证结果: {is_valid}")

    return 0.7*math_score + 0.3*semantic_score * depth_factor


def extract_answer(text):
    """从文本提取答案，增加有效性验证"""
    # 辅助函数验证数字格式
    def is_valid_number(s):
        return re.match(r"^[+-]?(?:\d+\.?\d*|\.\d+)$", s) is not None
    
    # 尝试匹配XML标签格式
    xml_match = re.search(r"<final_answer>(.*?)</final_answer>", text, re.IGNORECASE|re.DOTALL)
    if xml_match:
        content = xml_match.group(1).strip()
    else:
        content = text
    
    # 匹配带变量的答案模式
    var_pattern = r"""
        x\s*=\s*([+-]?\d+\.?\d*|\.\d+)  # 支持 .5 格式
        [,\s]+
        y\s*=\s*([+-]?\d+\.?\d*|\.\d+)
    """
    match = re.search(var_pattern, content, re.VERBOSE|re.IGNORECASE)
    
    if match:
        x_val, y_val = match.groups()
        if is_valid_number(x_val) and is_valid_number(y_val):
            return f"x={x_val},y={y_val}"
    
    # 备用模式：无变量名数值对
    num_pair_match = re.search(
        r"([+-]?(?:\d+\.?\d*|\.\d+))[,\s]+([+-]?(?:\d+\.?\d*|\.\d+))",
        content
    )
    if num_pair_match:
        x_val, y_val = num_pair_match.groups()
        if is_valid_number(x_val) and is_valid_number(y_val):
            return f"x={x_val},y={y_val}"
    
    return None


def tot_pipeline(problem):
    # 初始化根节点
    root = ToTNode(state=problem)
    
    # 首次扩展
    expand_node(root, beam_width=3)
    print("[DEBUG] 首次扩展后的子节点：")
    for i, child in enumerate(root.children):
        print(f" 子节点{i+1}：{child.to_dict()}")
    # MCTS优化
    root = monte_carlo_tree_search(root, iterations=50)
    # 调试输出根节点状态
    print(f"[DEBUG] 根节点属性: visits={root.visits}, value={root.value}")
    
    visualize_tree(root, filename="tot_structure")
    # 提取最优路径
    path = []
    node = root
    while node.children:
        node = max(node.children, key=lambda x: x.visits)
        path.append(node)
        if node.is_terminal:
            break
    
    # 结果处理
    if path and path[-1].is_terminal:
        final_answer = extract_answer(path[-1].state)
        if mathematical_verification(problem, final_answer):
            visualize_equations(problem, final_answer)
            return format_output(path, final_answer)
    
    # 错误处理
    return error_analysis(root)

def format_output(path, answer):
    """格式化输出结果"""
    steps = []
    for i, node in enumerate(path):
        step_info = f"Depth {node.depth}: {node.state.split('<solution>')[-1].split('</solution>')[0][:100]}"
        steps.append(f"{i+1}. {step_info}")
    return "\n".join([
        "="*40,
        "最优推导路径：",
        "\n".join(steps),
        "="*40,
        f"最终答案：{answer}",
        "="*40
    ])

def error_analysis(root):
    """错误分析"""
    terminals = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_terminal:
            terminals.append(node)
        stack.extend(node.children)
    
    error_info = [
        "="*40,
        "错误分析：",
        f"总终端节点数：{len(terminals)}",
        "候选答案分布："
    ]
    
    answer_counter = Counter()
    for node in terminals:
        ans = extract_answer(node.state)
        if ans:
            answer_counter[ans] += 1
    
    for ans, count in answer_counter.most_common():
        error_info.append(f"- {ans}: {count}票")
    
    return "\n".join(error_info)

from datetime import datetime
import time

def visualize_tree(root, filename="tot_structure", max_retries=3):
    """可视化树结构"""
    dot = Digraph(comment='ToT Structure')
    
    def add_nodes(node):
        node_id = str(hash(node.state))
        label = f"Depth {node.depth}\nVisits: {node.visits}\nValue: {node.value:.2f}"
        dot.node(node_id, label)
        for child in node.children:
            child_id = str(hash(child.state))
            dot.edge(node_id, child_id)
            add_nodes(child)
    
    add_nodes(root)
    
    # 生成唯一文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{filename}_{timestamp}"
    
    retries = 0
    while retries < max_retries:
        try:
            dot.render(unique_filename, view=False)  # 禁止自动打开
            print(f"树结构已保存至 {unique_filename}.pdf")
            return
        except PermissionError:
            print(f"文件被占用，重试中 ({retries+1}/{max_retries})...")
            time.sleep(2)
            retries += 1
    print("错误：无法写入文件，请手动关闭PDF阅读器后重试。")




# 数学验证层
def mathematical_verification(problem, answer):
    # 预处理答案
    answer = re.sub(r"[^0-9xy=.,-]", "", answer).replace(",", ",")
    try:
        problem_clean = re.sub(r"[\u2010-\u2015\-]", "-", problem)
        problem_clean = re.sub(r"解方程组：|方程\d+[:：]\s*", ";", problem_clean)
        problem_clean = re.sub(r"[^0-9xyab=+\-*/();]", " ", problem_clean)
        problem_clean = re.sub(r"(\d)([xyab])", r"\1*\2", problem_clean)
        problem_clean = re.sub(r"\s+", " ", problem_clean).strip()
        problem_clean = re.sub(r";+", ";", problem_clean).strip(';')

        equations = []
        equation_strings = [eq.strip() for eq in problem_clean.split(';') if eq.strip()]
        
        for eq_str in equation_strings:
            match = re.match(r"^\s*([^=]+?)\s*=\s*(.+?)\s*$", eq_str)
            if not match:
                print(f"方程格式错误: {eq_str}")
                return False
            lhs, rhs = match.groups()
            try:
                equations.append(Eq(sympify(lhs), sympify(rhs)))
            except Exception as e:
                print(f"方程解析失败: {lhs} = {rhs}，错误: {e}")
                return False
        
        answer_dict = {}
        valid_vars = {'x', 'y'}
        
        for pair in answer.split(','):
            if '=' not in pair:
                continue
                
            var_part, val_part = pair.split('=', 1)
            var = var_part.strip().lower()
            val = val_part.strip()
            
            # 新增数值预验证
            if not re.match(r"^[+-]?(?:\d+\.?\d*|\.\d+)$", val):
                print(f"无效数值格式: {val}")
                return False
                
            if var not in valid_vars:
                print(f"无效变量名: {var}")
                return False
                
            try:
                answer_dict[symbols(var)] = float(val)
            except ValueError:
                print(f"数值转换失败: {val}")
                return False
        
        if len(answer_dict) != 2:
            print("缺少必要变量")
            return False
            
        return all(eq.subs(answer_dict) for eq in equations)
    except Exception as e:
        print(f"验证错误: {str(e)}")
        return False
    

from sympy import lambdify

def visualize_equations(problem, solution=None):
    """绘制方程组图形"""
    try:
        # 解析方程
        x, y = symbols('x y')
        eqs = []
        problem_clean = re.sub(r"[\u2010-\u2015\-]", "-", problem)
        problem_clean = re.sub(r"解方程组：|方程\d+[:：]\s*", ";", problem_clean)
        problem_clean = re.sub(r"[^0-9xyab=+\-*/();]", " ", problem_clean)
        problem_clean = re.sub(r"(\d)([xyab])", r"\1*\2", problem_clean)
        problem_clean = re.sub(r"\s+", " ", problem_clean).strip()
        problem_clean = re.sub(r";+", ";", problem_clean).strip(';')
        # 分割方程并解析
        equation_strings = [eq.strip() for eq in problem_clean.split(';') if eq.strip()]
        for eq_str in equation_strings:
            match = re.match(r"^\s*([^=]+?)\s*=\s*(.+?)\s*$", eq_str)
            if not match:
                print(f"方程格式错误: {eq_str}")
                return False
            lhs, rhs = match.groups()
            try:
                eqs.append(Eq(sympify(lhs), sympify(rhs)))
            except Exception as e:
                print(f"方程解析失败: {lhs} = {rhs}，错误: {e}")
                return False

        # 生成坐标数据
        x_vals = np.linspace(-10, 10, 400)
        plt.figure(figsize=(8,6))
        colors = ['#1f77b4', '#ff7f0e']  # 不同方程的颜色
        
        for i, eq in enumerate(eqs):
            try:
                # 解方程获取y关于x的表达式
                y_expr = solve(eq, y)[0]
                y_func = lambdify(x, y_expr, modules='numpy')
                y_vals = y_func(x_vals)
                plt.plot(x_vals, y_vals, label=f'equation {i+1}', color=colors[i], lw=2)
            except:
                # 处理无法解析为y=f(x)的情况
                x_expr = solve(eq, x)[0]
                x_func = lambdify(y, x_expr, modules='numpy')
                y_vals = np.linspace(-10, 10, 400)
                x_vals = x_func(y_vals)
                plt.plot(x_vals, y_vals, label=f'equation {i+1}', color=colors[i], lw=2)

        # 标出解点
        if solution:
            x_val = float(solution.split('x=')[1].split(',')[0])
            y_val = float(solution.split('y=')[1])
            plt.scatter(x_val, y_val, color='red', s=100, 
                        zorder=5, label=f'Solution ({x_val}, {y_val})')
            plt.annotate(f'({x_val}, {y_val})', (x_val+0.5, y_val),
                         fontsize=10, color='darkred')

        plt.title("Visualization of systems of equations")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        
        # 保存图片或直接显示
        plt.savefig('equation_visualization.png', dpi=300)
        plt.close()
        print("可视化结果已保存至 equation_visualization.png")
        
    except Exception as e:
        print(f"可视化失败: {str(e)}")


def debug_answer_parsing():
    """答案解析调试函数"""
    test_cases = [
        ("<final_answer>x=2,y=4</final_answer>", True),
        ("Final_Answer: X=2.5, Y=3", True),
        ("答案：x=3, y=2.8", True),
        ("x=值,y=值", False),
        ("x=, y=4", False),
        ("无效内容", False)
    ]
    
    for text, expected in test_cases:
        result = extract_answer(text)
        success = (result is not None) == expected
        print(f"测试用例: {text[:30]}...")
        print(f"提取结果: {result} | 预期: {expected} | 状态: {'成功' if success else '失败'}")
        print("-"*60)

import torch
import torch.nn.functional as F
device='cuda'
# 困惑度计算函数
def compute_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    tokens = inputs["input_ids"]
    nll = F.nll_loss(log_probs[:, :-1].contiguous().view(-1, log_probs.size(-1)),
                     tokens[:, 1:].contiguous().view(-1),
                     reduction='mean')
    return torch.exp(nll).item()

if __name__ == "__main__":
    debug_answer_parsing()  # 添加测试
    test_cases = [
        "解方程组：\n方程1: 2x + 3y = 16\n方程2: x - 2y = -6",
    ]
    
    for problem in test_cases:
        print("\n" + "="*60)
        print(f"处理问题：{problem}")
        
        # 运行ToT流程
        root = ToTNode(state=problem)
        result = tot_pipeline(problem)
        print(result)
        
        # 生成可视化
        visualize_tree(root)
        print("="*60 + "\n")
