import os
from dotenv import load_dotenv
from evoagentx.workflow import WorkFlow, WorkFlowGraph
from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowEdge
from evoagentx.agents import CustomizeAgent
from evoagentx.prompts import StringTemplate
from evoagentx.models import OpenAILLMConfig
from evoagentx.tools.browser_use import BrowserUseToolkit
from evoagentx.tools.image_analysis import ImageAnalysisTool
from evoagentx.tools.flux_image_generation import FluxImageGenerationTool

load_dotenv()

# 需要的API密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BFL_API_KEY = os.getenv("BFL_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# 配置LLM
llm_config = OpenAILLMConfig(
    model="gpt-4o-mini", 
    openai_key=OPENAI_API_KEY, 
    stream=True, 
    output_response=True
)

def create_browser_research_agent():
    """
    Agent 1: Browser Research Agent with Image Analysis
    负责从网络获取热点信息和相关资料，并分析图片内容
    """
    
    # 创建工具
    browser_toolkit = BrowserUseToolkit(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        browser_type="chromium",
        headless=False
    )
    browser_tools = browser_toolkit.get_tools()
    
    # 添加图片分析工具
    image_analysis_tool = ImageAnalysisTool(
        api_key=OPENROUTER_API_KEY, 
        model="openai/gpt-4o-mini"
    )
    
    # 合并工具
    all_tools = browser_tools + [image_analysis_tool]
    
    research_agent = CustomizeAgent(
        name="BrowserResearchAgent",
        description="Web research agent with browser automation and image analysis capabilities",
        prompt_template=StringTemplate(
            instruction="""You are a professional web research assistant with both browser automation and image analysis capabilities.

Research Topic: {topic}

Please execute the following tasks:
1. Use browser tools to visit relevant websites (social media, news sites, forums)
2. Search for trending information related to the topic
3. If you encounter important images, use image analysis tool to understand their content
4. Collect comprehensive information including:
   - Current trending topics and hashtags
   - Popular discussions and opinions
   - Visual content insights (from image analysis)
   - Statistical data and metrics
   - User engagement patterns

5. Organize all findings into a structured research report

Requirements:
- Focus on social media trends and viral content
- Analyze both text and visual information
- Provide insights useful for content creation
- Include current engagement metrics when available"""
        ),
        llm_config=llm_config,
        inputs=[
            {"name": "topic", "type": "string", "description": "The topic to research"}
        ],
        outputs=[
            {"name": "research_info", "type": "string", "description": "Comprehensive research report with text and visual insights"}
        ],
        tools=all_tools
    )
    
    return research_agent

def create_content_generation_agent():
    """
    Agent 2: Content Generation Agent
    基于研究信息生成社交媒体推文内容
    """
    
    content_agent = CustomizeAgent(
        name="ContentGenerationAgent", 
        description="Social media content creation specialist",
        prompt_template=StringTemplate(
            instruction="""You are a social media content creation expert. Create engaging posts based on research data.

Research Information: {research_info}
Content Style: {style}
Target Platform: {platform}

Please create compelling social media content with:

1. **Hook**: Start with an attention-grabbing opening
2. **Main Content**: 
   - Use trending insights from research
   - Include relevant data points
   - Tell a story or share valuable information
   - Match the specified style (professional, casual, humorous, etc.)
3. **Engagement Elements**:
   - Ask questions to encourage interaction
   - Include relevant hashtags
   - Add call-to-action
4. **Platform Optimization**:
   - Adjust length for platform requirements
   - Use platform-specific features and formatting

Output Format:
- Main post text
- Suggested hashtags
- Engagement strategy notes

Requirements:
- Content must be original and engaging
- Use insights from the research data
- Optimize for the specified platform
- Match the requested style and tone"""
        ),
        llm_config=llm_config,
        inputs=[
            {"name": "research_info", "type": "string", "description": "Research information from previous step"},
            {"name": "style", "type": "string", "description": "Content style (professional, casual, humorous, etc.)"},
            {"name": "platform", "type": "string", "description": "Target social media platform"}
        ],
        outputs=[
            {"name": "post_content", "type": "string", "description": "Generated social media post content"}
        ]
    )
    
    return content_agent

def create_image_generation_agent():
    """
    Agent 3: Image Generation Agent
    基于内容生成配套的社交媒体图片
    """
    
    # 创建图片生成工具
    image_gen_tool = FluxImageGenerationTool(
        api_key=BFL_API_KEY, 
        save_path="./social_media_images"
    )
    
    image_agent = CustomizeAgent(
        name="ImageGenerationAgent",
        description="Social media image creation specialist",
        prompt_template=StringTemplate(
            instruction="""You are a visual content creator for social media. Generate compelling images based on post content.

Post Content: {post_content}
Research Context: {research_info}

Please create a detailed image generation prompt that will produce an engaging social media image:

1. **Visual Style**: Modern, eye-catching, social media optimized
2. **Content Elements**: 
   - Reflect the main theme of the post
   - Include visual metaphors or concepts from the content
   - Ensure it's visually appealing for social media feeds
3. **Technical Requirements**:
   - High contrast and vibrant colors
   - Clear composition that works at small sizes
   - No text elements (text will be added separately)
   - Aspect ratio suitable for social media

4. **Brand Consistency**: Professional yet approachable aesthetic

Generate a detailed English prompt for image creation that captures the essence of the social media post while being visually striking and platform-appropriate.

The prompt should be specific enough to create a cohesive visual that complements the written content."""
        ),
        llm_config=llm_config,
        inputs=[
            {"name": "post_content", "type": "string", "description": "Social media post content"},
            {"name": "research_info", "type": "string", "description": "Research context for visual inspiration"}
        ],
        outputs=[
            {"name": "image_path", "type": "string", "description": "Path to generated image file"}
        ],
        tools=[image_gen_tool]
    )
    
    return image_agent

def create_social_media_workflow():
    """
    创建完整的社交媒体内容生成工作流
    包含三个Agent：Research -> Content Generation -> Image Generation
    """
    
    # 1. 创建三个Agent
    research_agent = create_browser_research_agent()
    content_agent = create_content_generation_agent()
    image_agent = create_image_generation_agent()
    
    # 2. 创建工作流节点
    research_node = WorkFlowNode(name="research", module=research_agent)
    content_node = WorkFlowNode(name="content_generation", module=content_agent)
    image_node = WorkFlowNode(name="image_generation", module=image_agent)
    
    # 3. 定义工作流边（数据流向）
    edges = [
        # Research -> Content Generation
        WorkFlowEdge(
            source="research",
            target="content_generation",
            mappings={
                "research_info": "research_info",
                "style": "style",
                "platform": "platform"
            }
        ),
        # Content Generation -> Image Generation
        WorkFlowEdge(
            source="content_generation",
            target="image_generation", 
            mappings={
                "post_content": "post_content"
            }
        ),
        # Research -> Image Generation (为图片生成提供额外上下文)
        WorkFlowEdge(
            source="research",
            target="image_generation",
            mappings={
                "research_info": "research_info"
            }
        )
    ]
    
    # 4. 创建工作流图
    workflow_graph = WorkFlowGraph(
        nodes=[research_node, content_node, image_node],
        edges=edges
    )
    
    # 5. 创建完整工作流
    workflow = WorkFlow(
        name="SocialMediaContentWorkflow",
        description="Complete social media content creation workflow: Research -> Content -> Image",
        graph=workflow_graph,
        inputs=[
            {"name": "topic", "type": "string", "description": "Content topic to research"},
            {"name": "style", "type": "string", "description": "Content style (professional, casual, humorous, etc.)"},
            {"name": "platform", "type": "string", "description": "Target social media platform (Twitter, Instagram, LinkedIn, etc.)"}
        ],
        outputs=[
            {"name": "research_info", "type": "string", "description": "Research findings"},
            {"name": "post_content", "type": "string", "description": "Generated social media content"},
            {"name": "image_path", "type": "string", "description": "Path to generated image"}
        ]
    )
    
    return workflow

def test_complete_workflow():
    """
    测试完整的三节点工作流
    """
    # 检查API密钥
    required_keys = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "BFL_API_KEY": BFL_API_KEY,
        "OPENROUTER_API_KEY": OPENROUTER_API_KEY
    }
    
    missing_keys = [key for key, value in required_keys.items() if not value]
    if missing_keys:
        print("❌ 缺少必需的API密钥:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\n请在.env文件中添加所有必需的API密钥")
        return
    
    print("🚀 测试完整的社交媒体内容工作流")
    print("=" * 60)
    
    try:
        # 创建工作流
        workflow = create_social_media_workflow()
        print("✅ 完整工作流创建成功!")
        
        # 测试案例
        test_cases = [
            {
                "topic": "AI productivity tools for professionals",
                "style": "professional",
                "platform": "LinkedIn"
            },
            {
                "topic": "Healthy morning routines",
                "style": "casual",
                "platform": "Instagram"
            },
            {
                "topic": "Remote work productivity tips",
                "style": "humorous",
                "platform": "Twitter"
            }
        ]
        
        print("\n🎯 可选测试案例：")
        for i, case in enumerate(test_cases, 1):
            print(f"{i}. {case['topic']} ({case['style']} style for {case['platform']})")
        
        print("\n请选择测试案例（输入数字1-3）:")
        try:
            choice = int(input().strip())
            if 1 <= choice <= len(test_cases):
                selected_case = test_cases[choice - 1]
                print(f"\n🎯 执行案例: {selected_case['topic']}")
                print(f"   风格: {selected_case['style']}")
                print(f"   平台: {selected_case['platform']}")
                
                # 执行完整工作流
                result = workflow.run(**selected_case)
                
                print("\n" + "="*60)
                print("🎉 工作流执行完成！")
                print("="*60)
                
                print("\n📊 1. 研究结果:")
                print("-" * 40)
                print(result['research_info'])
                
                print("\n📝 2. 生成的内容:")
                print("-" * 40)
                print(result['post_content'])
                
                print(f"\n🖼️ 3. 生成的图片:")
                print("-" * 40)
                print(f"图片路径: {result['image_path']}")
                
                # 尝试显示图片
                if os.path.exists(result['image_path']):
                    try:
                        from PIL import Image
                        img = Image.open(result['image_path'])
                        img.show()
                        print("✅ 图片已自动打开")
                    except ImportError:
                        print("💡 安装PIL查看图片: pip install Pillow")
                    except Exception as e:
                        print(f"❌ 显示图片失败: {e}")
                else:
                    print("❌ 图片文件未找到")
                
                return result
                
            else:
                print("❌ 无效选择")
                
        except (ValueError, KeyboardInterrupt):
            print("\n💡 你也可以直接运行:")
            print('result = workflow.run(topic="your topic", style="professional", platform="LinkedIn")')
            
    except Exception as e:
        print(f"❌ 工作流执行失败: {e}")
        print("\n💡 可能的问题:")
        print("1. 依赖安装: pip install browser-use")
        print("2. 浏览器: 需要Chrome/Chromium")
        print("3. 网络连接问题")
        print("4. API密钥配置错误")

def test_direct_agent():
    """
    直接测试单个Agent（不通过workflow）
    """
    print("🧪 直接测试Browser Research Agent")
    print("=" * 40)
    
    try:
        agent = create_browser_research_agent()
        
        # 简单测试
        topic = "Today's trending topics"
        print(f"🎯 测试主题: {topic}")
        
        result = agent(inputs={"topic": topic})
        
        print("\n📋 Agent直接输出:")
        print(result.content.research_info)
        
    except Exception as e:
        print(f"❌ 直接测试失败: {e}")

if __name__ == "__main__":
    """
    🔧 配置说明：
    
    完整的三节点社交媒体内容工作流
    
    1. 必需环境变量：
       - OPENAI_API_KEY: OpenAI API密钥 (用于LLM和浏览器工具)
       - BFL_API_KEY: Black Forest Labs API密钥 (用于图片生成)
       - OPENROUTER_API_KEY: OpenRouter API密钥 (用于图片分析)
    
    2. 依赖安装：
       - pip install browser-use (Python 3.11+)
       - pip install browser-use-py310x (Python 3.10)
       - pip install Pillow (用于图片显示)
    
    3. 工作流节点：
       - Node 1: Browser Research Agent (browser + image analysis)
       - Node 2: Content Generation Agent (创建推文内容)
       - Node 3: Image Generation Agent (生成配图)
    
    4. 数据流向：
       Research -> Content Generation -> Image Generation
       Research -> Image Generation (提供额外上下文)
    """
    
    print("📱 完整社交媒体内容工作流")
    print("三节点工作流：Research -> Content -> Image")
    print("=" * 60)
    
    # 运行完整工作流测试
    # test_complete_workflow()
    
    # 可选：直接测试单个Agent
    test_direct_agent() 