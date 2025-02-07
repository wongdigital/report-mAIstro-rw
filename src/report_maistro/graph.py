from langchain_anthropic import ChatAnthropic 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
import asyncio

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph

from src.report_maistro.state import ReportStateInput, ReportStateOutput, Sections, ReportState, SectionState, SectionOutputState, Queries
from src.report_maistro.prompts import report_planner_query_writer_instructions, report_planner_instructions, query_writer_instructions, section_writer_instructions, final_section_writer_instructions
from src.report_maistro.configuration import Configuration
from src.report_maistro.utils import tavily_search_async, deduplicate_and_format_sources, format_sections

# LLMs 
planner_model = ChatOpenAI(model=Configuration.planner_model, reasoning_effort="medium") 
writer_model = ChatAnthropic(model=Configuration.writer_model, temperature=0) 

# Nodes
async def generate_report_plan(state: ReportState, config: RunnableConfig):
    """ Generate the report plan """

    # Inputs
    topic = state["topic"]
    feedback = state.get("feedback_on_report_plan", None)

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    tavily_topic = configurable.tavily_topic
    tavily_days = configurable.tavily_days

    # Convert JSON object to string if necessary
    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    # Generate search query
    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions_query = report_planner_query_writer_instructions.format(topic=topic, report_organization=report_structure, number_of_queries=number_of_queries)

    # Generate queries  
    results = structured_llm.invoke([SystemMessage(content=system_instructions_query)]+[HumanMessage(content="Generate search queries that will help with planning the sections of the report.")])

    # Web search
    query_list = [query.search_query for query in results.queries]

    # Search web 
    search_docs = await tavily_search_async(query_list, tavily_topic, tavily_days)

    # Deduplicate and format sources
    source_str = deduplicate_and_format_sources(search_docs, max_tokens_per_source=1000, include_raw_content=False)

    # Format system instructions
    system_instructions_sections = report_planner_instructions.format(topic=topic, report_organization=report_structure, context=source_str, feedback=feedback)

    # Generate sections 
    structured_llm = planner_model.with_structured_output(Sections)
    report_sections = structured_llm.invoke([SystemMessage(content=system_instructions_sections)]+[HumanMessage(content="Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. Each section must have: name, description, plan, research, and content fields.")])

    return {"sections": report_sections.sections}

def human_feedback(state: ReportState):
    """ No-op node that should be interrupted on """
    pass

def generate_queries(state: SectionState, config: RunnableConfig):
    """ Generate search queries for a report section """

    # Get state 
    section = state["section"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # Generate queries 
    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions = query_writer_instructions.format(section_topic=section.description, number_of_queries=number_of_queries)

    # Generate queries  
    queries = structured_llm.invoke([SystemMessage(content=system_instructions)]+[HumanMessage(content="Generate search queries on the provided topic.")])

    return {"search_queries": queries.queries}

async def search_web(state: SectionState, config: RunnableConfig):
    """ Search the web for each query, then return a list of raw sources and a formatted string of sources."""
    
    # Get state 
    search_queries = state["search_queries"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    tavily_topic = configurable.tavily_topic
    tavily_days = configurable.tavily_days

    # Web search
    query_list = [query.search_query for query in search_queries]
    search_docs = await tavily_search_async(query_list, tavily_topic, tavily_days)

    # Deduplicate and format sources
    source_str = deduplicate_and_format_sources(search_docs, max_tokens_per_source=5000, include_raw_content=True)

    return {"source_str": source_str}

def write_section(state: SectionState):
    """ Write a section of the report """

    # Get state 
    section = state["section"]
    source_str = state["source_str"]

    # Limit the source string size to reduce token count while preserving complete sentences
    limited_sources = []
    for source in source_str.split("\n\nSource:"):
        if not source.strip():
            continue
            
        # Split source into sentences (roughly)
        sentences = [s.strip() + "." for s in source.replace("\n", " ").split(".") if s.strip()]
        
        # Calculate rough token count (4 chars ~= 1 token)
        total_chars = 0
        limited_source = []
        
        # Keep sentences up to roughly 1000 tokens (4000 chars), preserving complete sentences
        for sentence in sentences:
            sentence_chars = len(sentence)
            if total_chars + sentence_chars > 4000:
                break
            limited_source.append(sentence)
            total_chars += sentence_chars
            
        if limited_source:
            # Reconstruct source with complete sentences
            limited_sources.append(f"Source:{' '.join(limited_source)}")
    
    # Join all limited sources
    limited_source_str = "\n\n".join(limited_sources)

    # Format system instructions
    system_instructions = section_writer_instructions.format(section_title=section.name, section_topic=section.description, context=limited_source_str)

    # Generate section  
    section_content = writer_model.invoke([SystemMessage(content=system_instructions)]+[HumanMessage(content="Generate a report section based on the provided sources.")])
    
    # Write content to the section object  
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}

def initiate_section_writing(state: ReportState):
    """ This is the "map" step when we kick off web research for some sections of the report """    
        
    # Get feedback
    feedback = state.get("feedback_on_report_plan", None)

    # Feedback is by default None and accept_report_plan is by default False
    # If a user hits "Continue" in Studio, we want to proceed with the report plan
    # If a user enters feedback_on_report_plan in Studio, we want to regenerate the report plan
    # Once a user enters feedback_on_report_plan, they need to flip accept_report_plan to True to proceed
    if not state.get("accept_report_plan") and feedback:
        return "generate_report_plan"
    
    # Kick off section writing in parallel via Send() API for any sections that require research
    else: 
        return [
            Send("build_section_with_web_research", {"section": s}) 
            for s in state["sections"] 
            if s.research
        ]

async def write_final_sections(state: SectionState):
    """ Write final sections of the report, which do not require web search and use the completed sections as context """

    # Add delay to avoid rate limits
    await asyncio.sleep(2)  # 2 second delay

    # Get state 
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    
    # Limit context size by taking only the most relevant sections
    # For introduction, we want a brief overview of all sections
    # For conclusion, we want detailed content from all sections
    max_tokens_per_section = 2000 if section.name.lower() == "conclusion" else 500
    
    # Split into sections and limit each section's size
    sections = completed_report_sections.split("="*60)
    limited_sections = []
    for s in sections:
        if len(s.strip()) > 0:  # Skip empty sections
            # Keep section header and first part up to max_tokens_per_section
            section_parts = s.split("\n", 3)  # Split into: newline, title, newline, content
            if len(section_parts) >= 4:
                limited_content = section_parts[3][:max_tokens_per_section]
                limited_sections.append(f"\n{'='*60}{section_parts[1]}\n{'='*60}\n{limited_content}")
    
    limited_context = "\n".join(limited_sections)
    
    # Format system instructions
    system_instructions = final_section_writer_instructions.format(section_title=section.name, section_topic=section.description, context=limited_context)

    # Generate section  
    section_content = writer_model.invoke([SystemMessage(content=system_instructions)]+[HumanMessage(content="Generate a report section based on the provided sources.")])
    
    # Write content to section 
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}

def gather_completed_sections(state: ReportState):
    """ Gather completed sections from research and format them as context for writing the final sections """    

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(completed_sections)

    return {"report_sections_from_research": completed_report_sections}

def initiate_final_section_writing(state: ReportState):
    """ Write any final sections sequentially to avoid rate limits """    

    # Get all sections that don't require research
    non_research_sections = [s for s in state["sections"] if not s.research]
    
    # Get names of completed sections
    completed_section_names = {s.name for s in state["completed_sections"]}
    
    # Find the first unwritten non-research section
    for section in non_research_sections:
        if section.name not in completed_section_names:
            return Send("write_final_sections", {"section": section, "report_sections_from_research": state["report_sections_from_research"]})
    
    # If all sections are written, move to final compilation
    return "compile_final_report"

def compile_final_report(state: ReportState):
    """ Compile the final report """    

    # Get sections and completed sections
    sections = state["sections"]
    completed_sections = state["completed_sections"]
    
    # Create a map of section name to content
    section_content_map = {s.name: s.content for s in completed_sections}
    
    # Verify all sections are completed
    missing_sections = []
    for section in sections:
        if section.name not in section_content_map:
            missing_sections.append(section.name)
    
    if missing_sections:
        raise ValueError(f"Missing content for sections: {', '.join(missing_sections)}")

    # Update sections with completed content while maintaining original order
    final_sections = []
    for section in sections:
        section.content = section_content_map[section.name]
        final_sections.append(section.content)

    # Compile final report
    all_sections = "\n\n".join(final_sections)

    return {"final_report": all_sections}

# Report section sub-graph -- 

# Add nodes 
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")
section_builder.add_edge("write_section", END)

# Outer graph -- 

# Add nodes
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# Add edges
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_conditional_edges("human_feedback", initiate_section_writing, ["build_section_with_web_research", "generate_report_plan"])
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections", "compile_final_report"])
builder.add_edge("write_final_sections", "gather_completed_sections")
builder.add_edge("compile_final_report", END)

graph = builder.compile(interrupt_before=['human_feedback'])