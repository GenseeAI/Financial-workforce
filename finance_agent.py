import json
import os
import yfinance as yf
import asyncio
import pandas as pd
import io
import sys
import time
from datetime import datetime, timedelta
from camel.toolkits.function_tool import FunctionTool
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.societies.workforce import Workforce
from camel.tasks import Task
from camel.types import RoleType


class AgentResponseEncoder(json.JSONEncoder):
    """Custom JSON encoder for agent responses containing complex objects."""
    
    def default(self, obj):
        if isinstance(obj, BaseMessage):
            return {
                "role_name": obj.role_name,
                "role_type": obj.role_type.name if hasattr(obj.role_type, 'name') else str(obj.role_type),
                "content": obj.content,
                "meta_dict": obj.meta_dict,
                "video_bytes": None,  # Skip video bytes to avoid serialization issues
                "image_list": obj.image_list,
                "image_detail": obj.image_detail,
                "video_detail": obj.video_detail,
                "parsed": obj.parsed
            }
        elif isinstance(obj, RoleType):
            return obj.name
        elif hasattr(obj, '__dict__'):
            # For any other object with attributes, convert to dict
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        # For objects that can't be serialized, convert to string
        return str(obj)


def convert_agent_response_to_dict(agent_response):
    """Convert agent response to a JSON-serializable dictionary."""
    try:
        # Handle the agent response which has msgs, terminated, and info attributes
        result = {
            "messages": [],
            "terminated": agent_response.terminated if hasattr(agent_response, 'terminated') else False,
            "info": {}
        }
        
        # Convert messages
        if hasattr(agent_response, 'msgs') and agent_response.msgs:
            for msg in agent_response.msgs:
                if isinstance(msg, BaseMessage):
                    result["messages"].append({
                        "role_name": msg.role_name,
                        "role_type": msg.role_type.name if hasattr(msg.role_type, 'name') else str(msg.role_type),
                        "content": msg.content,
                        "meta_dict": msg.meta_dict,
                        "video_bytes": None,  # Skip to avoid serialization issues
                        "image_list": msg.image_list,
                        "image_detail": msg.image_detail,
                        "video_detail": msg.video_detail,
                        "parsed": msg.parsed
                    })
                else:
                    result["messages"].append(str(msg))
        
        # Convert info
        if hasattr(agent_response, 'info') and agent_response.info:
            result["info"] = {}
            for key, value in agent_response.info.items():
                try:
                    # Try to serialize the value to check if it's JSON serializable
                    json.dumps(value)
                    result["info"][key] = value
                except (TypeError, ValueError):
                    # If not serializable, convert to string
                    result["info"][key] = str(value)
        
        return result
        
    except Exception as e:
        # Fallback: return a basic representation
        return {
            "messages": [str(agent_response)] if agent_response else [],
            "terminated": False,
            "info": {"conversion_error": str(e)},
            "raw_response": str(agent_response)
        }


def get_current_stock_price(ticker: str) -> str:
    """Get current stock price and basic info."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current market data - try to get real-time data first
        try:
            # Get the most recent data available
            hist = stock.history(period="1d", interval="1m")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                current_volume = hist['Volume'].iloc[-1]
                current_time = hist.index[-1]
            else:
                # Fallback to daily data
                hist = stock.history(period="1d")
                if hist.empty:
                    return f"Could not retrieve data for {ticker}"
                current_price = hist['Close'].iloc[-1]
                current_volume = hist['Volume'].iloc[-1]
                current_time = hist.index[-1]
        except:
            # Final fallback to basic info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            current_volume = info.get('volume', 0)
            current_time = datetime.now()
        
        # Get previous close for change calculation
        prev_close = info.get('previousClose', current_price)
        change_amount = current_price - prev_close
        change_pct = (change_amount / prev_close * 100) if prev_close and prev_close != 0 else 0
        
        # Get additional current market info
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', 0)
        
        result = {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "previous_close": round(prev_close, 2),
            "change_amount": round(change_amount, 2),
            "change_percent": round(change_pct, 2),
            "current_volume": int(current_volume) if current_volume else 0,
            "market_cap": int(market_cap) if market_cap else 0,
            "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
            "company": info.get('longName', ticker),
            "sector": info.get('sector', 'Unknown'),
            "timestamp": current_time.isoformat() if hasattr(current_time, 'isoformat') else str(current_time)
        }

        return json.dumps(result)
    except Exception as e:
        return f"Error getting current stock price for {ticker}: {str(e)}"

def get_historical_stock_data(ticker: str, time_frame: str = "3mo") -> str:
    """Get stock data for a specified time frame for trend analysis. 
        The time frame options are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
        The ticker is the stock ticker you are looking for.
    """
    try:
        stock = yf.Ticker(ticker)
        # Get data for specified time frame
        hist = stock.history(period=time_frame)
        
        if hist.empty:
            return f"Could not retrieve {time_frame} data for {ticker}"
        
        # Calculate key metrics
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        high_price = hist['High'].max()
        low_price = hist['Low'].min()
        
        avg_volume = hist['Volume'].mean()
        # Calculate trend (percentage change)
        trend = ((end_price - start_price) / start_price * 100)
        
        # Calculate volatility
        volatility = hist['Close'].std()
        
        result = {
            "ticker": ticker,
            "period": time_frame,
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "high_price": round(high_price, 2),
            "low_price": round(low_price, 2),
            "trend": round(trend, 2),
            "volatility": round(volatility, 2),
            "avg_volume": int(avg_volume) if avg_volume else 0
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error getting {time_frame} data: {str(e)}"




def get_current_stock_price(ticker: str) -> str:
    """Get current stock price and basic info."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current market data - try to get real-time data first
        try:
            # Get the most recent data available
            hist = stock.history(period="1d", interval="1m")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                current_volume = hist['Volume'].iloc[-1]
                current_time = hist.index[-1]
            else:
                # Fallback to daily data
                hist = stock.history(period="1d")
                if hist.empty:
                    return f"Could not retrieve data for {ticker}"
                current_price = hist['Close'].iloc[-1]
                current_volume = hist['Volume'].iloc[-1]
                current_time = hist.index[-1]
        except:
            # Final fallback to basic info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            current_volume = info.get('volume', 0)
            current_time = datetime.now()
        
        # Get previous close for change calculation
        prev_close = info.get('previousClose', current_price)
        change_amount = current_price - prev_close
        change_pct = (change_amount / prev_close * 100) if prev_close and prev_close != 0 else 0
        
        # Get additional current market info
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', 0)
        
        result = {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "previous_close": round(prev_close, 2),
            "change_amount": round(change_amount, 2),
            "change_percent": round(change_pct, 2),
            "current_volume": int(current_volume) if current_volume else 0,
            "market_cap": int(market_cap) if market_cap else 0,
            "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
            "company": info.get('longName', ticker),
            "sector": info.get('sector', 'Unknown'),
            "timestamp": current_time.isoformat() if hasattr(current_time, 'isoformat') else str(current_time)
        }

        return json.dumps(result)
    except Exception as e:
        return f"Error getting current stock price for {ticker}: {str(e)}"

def get_historical_stock_data(ticker: str, time_frame: str = "3mo") -> str:
    """Get stock data for a specified time frame for trend analysis. 
        The time frame options are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
        The ticker is the stock ticker you are looking for.
    """
    try:
        stock = yf.Ticker(ticker)
        # Get data for specified time frame
        hist = stock.history(period=time_frame)
        
        if hist.empty:
            return f"Could not retrieve {time_frame} data for {ticker}"
        
        # Calculate key metrics
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        high_price = hist['High'].max()
        low_price = hist['Low'].min()
        
        avg_volume = hist['Volume'].mean()
        # Calculate trend (percentage change)
        trend = ((end_price - start_price) / start_price * 100)
        
        # Calculate volatility
        volatility = hist['Close'].std()
        
        result = {
            "ticker": ticker,
            "period": time_frame,
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "high_price": round(high_price, 2),
            "low_price": round(low_price, 2),
            "trend": round(trend, 2),
            "volatility": round(volatility, 2),
            "avg_volume": int(avg_volume) if avg_volume else 0
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error getting {time_frame} data: {str(e)}"

def extract_subtask_questions_from_stdout(stdout_text):
    """
    Extract subtask questions from the stdout logs.
    """
    subtask_questions = {}
    lines = stdout_text.split('\n')
    
    for i, line in enumerate(lines):
        if 'get task' in line and ':' in line:
            # Extract task ID and question
            parts = line.split('get task')
            if len(parts) > 1:
                task_part = parts[1].strip()
                if ':' in task_part:
                    task_id, question = task_part.split(':', 1)
                    task_id = task_id.strip()
                    question = question.strip()
                    subtask_questions[task_id] = question
    
    return subtask_questions

def format_task_results_with_questions(task):
    """
    Format the task results to include both subtask questions and their answers.
    """
    try:
        # Get the original question
        original_question = task.content
        
        # Initialize result structure
        formatted_output = f"Original Question: {original_question}\n\n"
        formatted_output += "=" * 50 + "\n\n"
        
        # Try to extract subtask information from the result
        if hasattr(task, 'result') and task.result:
            result_text = str(task.result)
            
            # Check if the result contains subtask information
            if "--- Subtask" in result_text:
                # Parse the existing subtask results
                result_lines = result_text.split('\n')
                current_subtask = None
                current_answer = []
                
                for line in result_lines:
                    line = line.strip()
                    if line.startswith("--- Subtask"):
                        # If we have a previous subtask, add it to output
                        if current_subtask and current_answer:
                            formatted_output += f"{current_subtask}\n"
                            formatted_output += f"Answer: {' '.join(current_answer).strip()}\n\n"
                        
                        # Start new subtask
                        current_subtask = line
                        current_answer = []
                    elif line and current_subtask:
                        # This is part of the answer for the current subtask
                        current_answer.append(line)
                    elif line and not current_subtask:
                        # This might be a standalone result
                        formatted_output += f"Result: {line}\n\n"
                
                # Don't forget the last subtask
                if current_subtask and current_answer:
                    formatted_output += f"{current_subtask}\n"
                    formatted_output += f"Answer: {' '.join(current_answer).strip()}\n\n"
                    
            else:
                # If no subtask structure, just show the result
                formatted_output += f"Final Result: {task.result}\n"
        else:
            formatted_output += "No results available.\n"
        
        return formatted_output
        
    except Exception as e:
        # Fallback to original result if formatting fails
        return f"Original Question: {task.content}\n\nResult: {task.result}\n\nError formatting results: {str(e)}"

def format_complete_results_with_questions(task, stdout_text=""):
    """
    Format complete results including subtask questions extracted from stdout.
    """
    try:
        # Get the original question
        original_question = task.content
        
        # Initialize result structure
        formatted_output = f"Original Question: {original_question}\n\n"
        formatted_output += "=" * 60 + "\n\n"
        
        # Extract subtask questions from stdout if available
        subtask_questions = {}
        if stdout_text:
            subtask_questions = extract_subtask_questions_from_stdout(stdout_text)
        
        # Try to extract subtask information from the result
        if hasattr(task, 'result') and task.result:
            result_text = str(task.result)
            
            # Check if the result contains subtask information
            if "--- Subtask" in result_text:
                # Parse the existing subtask results
                result_lines = result_text.split('\n')
                current_subtask = None
                current_answer = []
                
                for line in result_lines:
                    line = line.strip()
                    if line.startswith("--- Subtask"):
                        # If we have a previous subtask, add it to output
                        if current_subtask and current_answer:
                            subtask_id = current_subtask.replace("--- Subtask ", "").split(" ")[0]
                            question = subtask_questions.get(subtask_id, "Question not available")
                            formatted_output += f"{current_subtask}\n"
                            formatted_output += f"Question: {question}\n"
                            formatted_output += f"Answer: {' '.join(current_answer).strip()}\n\n"
                        
                        # Start new subtask
                        current_subtask = line
                        current_answer = []
                    elif line and current_subtask:
                        # This is part of the answer for the current subtask
                        current_answer.append(line)
                    elif line and not current_subtask:
                        # This might be a standalone result
                        formatted_output += f"Result: {line}\n\n"
                
                # Don't forget the last subtask
                if current_subtask and current_answer:
                    subtask_id = current_subtask.replace("--- Subtask ", "").split(" ")[0]
                    question = subtask_questions.get(subtask_id, "Question not available")
                    formatted_output += f"{current_subtask}\n"
                    formatted_output += f"Question: {question}\n"
                    formatted_output += f"Answer: {' '.join(current_answer).strip()}\n\n"
                    
            else:
                # If no subtask structure, just show the result
                formatted_output += f"Final Result: {task.result}\n"
        else:
            formatted_output += "No results available.\n"
    
        return formatted_output
        
    except Exception as e:
        # Fallback to original result if formatting fails
        return f"Original Question: {task.content}\n\nResult: {task.result}\n\nError formatting results: {str(e)}"

def create_comprehensive_output(task, stdout_text="", include_stdout=True):
    """
    Create a comprehensive output including formatted results and optionally original stdout.
    
    Args:
        task: The task object containing results
        stdout_text: Captured stdout text
        include_stdout: Whether to include the full stdout logs
    """
    try:
        # Get the original question
        original_question = task.content
        
        # Create the main formatted output
        main_output = format_complete_results_with_questions(task, stdout_text)
        
        if include_stdout:
            # Add the original stdout for transparency
            comprehensive_output = main_output + "\n" + "=" * 60 + "\n"
            comprehensive_output += "ORIGINAL STDOUT LOGS (for debugging):\n"
            comprehensive_output += "=" * 60 + "\n"
            comprehensive_output += stdout_text if stdout_text else "No stdout captured"
        else:
            comprehensive_output = main_output
        
        return comprehensive_output
        
    except Exception as e:
        return f"Error creating comprehensive output: {str(e)}\n\nOriginal Result: {task.result}"
def single_agent(input_question: str, tools: list) -> dict:
    agent = ChatAgent(
        system_message = BaseMessage.make_assistant_message(
            role_name="Financial Analyst",
            content="""
                You are a financial analyst.
                You are given a question and you need to answer it.
                You have access to the following tools: 
                - get_current_stock_price: Get the current stock price and basic info
                - get_historical_stock_data: Get stock data for a specified time frame for trend analysis. The time frame options are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max. 
                - tavily_search_sync: Search the internet using Tavily API
                - ggl_search_sync: Search using Google Custom Search API
                - exa_search_sync: Search using Exa Search API
                - scale_search_sync: Use multiple search engines for robust web searching
            """,
        ),
        model = ModelFactory.create(
            model_platform="openai",  
            model_type="gpt-5",   
        ),
        tools=tools,
    )
    task = agent.step(input_question)
    # Convert the agent response to a JSON-serializable format
    return convert_agent_response_to_dict(task)

def multi_agent_with_verifier(input_question: str, tools: list, include_stdout: bool = True) -> dict:
    agent = ChatAgent(
        system_message = BaseMessage.make_assistant_message(
            role_name="Financial Analyst",
            content="""
                You are a financial analyst.
                You are given a question and you need to answer it.
                You have access to the following tools: 
                - get_current_stock_price: Get the current stock price and basic info
                - get_historical_stock_data: Get stock data for a specified time frame for trend analysis. The time frame options are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max. 
                - tavily_search_sync: Search the internet using Tavily API
                - ggl_search_sync: Search using Google Custom Search API
                - exa_search_sync: Search using Exa Search API
                - scale_search_sync: Use multiple search engines for robust web searching
            """,
        ),
        model = ModelFactory.create(
            model_platform="openai",  
            model_type="gpt-4o-mini",   
        ),
        tools=tools,
    )

    coordinator = ChatAgent(
        system_message = BaseMessage.make_assistant_message(
            role_name="Coordinator",
            content="""
            Your job is to coordinate the workforce and assign tasks to workers.
            """,
        ),
        model = ModelFactory.create(
            model_platform="openai",
            model_type="gpt-4o-mini",
        ),
    )
    planner = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Task Planner",
            content="Your job is to decompose tasks and compose results."
        ),
        model = ModelFactory.create(
            model_platform="openai",
            model_type="gpt-4o-mini",
        ),
    )
        
    workforce = Workforce(
        "Financial Analyst",
        coordinator_agent = coordinator,
        task_agent = planner,
    )

    workforce.add_single_agent_worker(
        description = "worker agent for financial analysis",
        worker = agent,
    )

    task = Task(
        content = input_question,
        id="0",
    )


    # Capture stdout to extract subtask questions and next steps
    start_ts = time.time()
    old_stdout = sys.stdout
    stdout_capture = io.StringIO()
    sys.stdout = stdout_capture
    try:
        processed_task = workforce.process_task(task)
        captured_stdout = stdout_capture.getvalue()
    finally:
        sys.stdout = old_stdout
        stdout_capture.close()

    end_ts = time.time()

    # Build detailed outputs
    final_result_str = str(getattr(processed_task, 'result', ''))
    subtasks = extract_subtask_questions_from_stdout(captured_stdout)
    comprehensive_results = create_comprehensive_output(processed_task, captured_stdout, include_stdout)

    return {
        "final_result": final_result_str,
        "stdout": captured_stdout,
        "subtask_questions": subtasks,
        "comprehensive_output": comprehensive_results,
        "timing_ms": int((end_ts - start_ts) * 1000),
    }

def app(input_question: str, output_json_path: str = \
    "/Users/allen/Desktop/demo_agents/Finance Agent 2/test_tool_results.json") -> dict:

    test_tools = [tavily_search_sync, ggl_search_sync, exa_search_sync, scale_search_sync]
    # test_tools = [tavily_search_sync, ggl_search_sync, scale_search_sync]
    results_by_tool = {}

    for tool_fn in test_tools:
        print(f"Running {tool_fn.__name__}")
        tool_name = getattr(tool_fn, "__name__", str(tool_fn))
        tools = [
            # FunctionTool(get_current_stock_price),
            # FunctionTool(get_historical_stock_data),
            FunctionTool(tool_fn),
        ]

        # details = multi_agent_with_verifier(input_question, tools, True)
        details = single_agent(input_question, tools)
        # Prepare a concise JSON-friendly record
        record = {
            "invoke": {
                "input_question": input_question,
                "agent_tools": [
                    # "get_current_stock_price",
                    # "get_historical_stock_data",
                    tool_name,
                ],
                "models": {
                    "agent": "gpt-5"
                }
            },
            "response": details,
            # "response": details.get("final_result", ""),
            # "next_steps": list(details.get("subtask_questions", {}).values()),
            # "stdout": details.get("stdout", ""),
            # "comprehensive_output": details.get("comprehensive_output", ""),
            # "timing_ms": details.get("timing_ms"),
        }

        # If scale search was used, capture which engines produced the results
        if tool_name == "scale_search_sync":
            try:
                raw_results = tool_fn(input_question)
                engines_used = set()
                if isinstance(raw_results, list):
                    for item in raw_results:
                        if isinstance(item, dict):
                            engines = item.get("engines")
                            engine = item.get("engine")
                            if isinstance(engines, list):
                                for eng in engines:
                                    if isinstance(eng, str) and eng:
                                        engines_used.add(eng)
                            elif isinstance(engine, str) and engine:
                                engines_used.add(engine)
                record["engines_used"] = sorted(list(engines_used))
            except Exception as e:
                record["engines_used_error"] = str(e)

        results_by_tool[tool_name] = record

    # Write to JSON file keyed by tool name
    try:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results_by_tool, f, ensure_ascii=False, indent=2, cls=AgentResponseEncoder)
    except Exception as e:
        # If writing fails, still return the data in-memory
        results_by_tool["__write_error__"] = str(e)

    return results_by_tool

def main():
    questions = [
        """By how much did NVIDIA's GAAP Research and Development expense exceed AMD's in fiscal year 2024, using only the "Research and development" line item from each company's latest Form 10-K filed with the SEC? Answer with a single number in USD billions, rounded to one decimal.""",
        """Using each company's 2024 sustainability/ESG report, what is the combined total of Scope 3 Category 11 ("Use of sold products") greenhouse gas emissions reported by Shell plc and BP p.l.c.? Answer with a single number in million metric tons CO2e, rounded to one decimal.""",
        """Using each company's 2024 Form 20-F, what is the combined revenue in USD for Baidu, Inc. and JD.com, Inc., computed by converting each company's reported total revenues in RMB using the average exchange rate for 2024 disclosed in its own filing (do not use end-of-period or convenience translation rates). Answer with a single number in USD billions, rounded to one decimal.""",
         ]
    results = []
    for idx, question in enumerate(questions):
        print(f"Running {question}")
        result = app(question, output_json_path=f"test_tool_results_{idx}.json")
        print(result)
        results.append(result)
        print("-"*100)
    return results


if __name__ == "__main__":
    main()