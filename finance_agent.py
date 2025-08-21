import json
import os
import yfinance as yf
import asyncio
import pandas as pd
import io
import sys
import time
from datetime import datetime, timedelta
from tools import TavilySearchTool
from camel.toolkits.function_tool import FunctionTool
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.societies.workforce import Workforce
from camel.tasks import Task
from camel.types import RoleType


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

def multi_agent_with_verifier(input_question: str, include_stdout: bool = True) -> dict:
    search_tools = [FunctionTool(TavilySearchTool.search_internet)]
    
    search_agent = ChatAgent(
        system_message = BaseMessage.make_assistant_message(
            role_name="Financial Analyst",
            content="""
                You are a search agent.
                You are given a question and you need to answer it using the tools provided.
                You have access to the following tools: 
                - TavilySearchTool.search_internet: Search the internet using Tavily API
            """,
        ),
        model = ModelFactory.create(
            model_platform="openai",  
            model_type="gpt-4o",   
        ),
        tools=search_tools,
    )
    stock_tools = [FunctionTool(get_current_stock_price), FunctionTool(get_historical_stock_data)]
    stock_agent = ChatAgent(
        system_message = BaseMessage.make_assistant_message(
            role_name="Stock Agent",
            content="""
                You are a stock agent.
                You are given a question and you need to answer it using the tools provided.
                You have access to the following tools: 
                - get_current_stock_price: Get the current stock price and basic info
                - get_historical_stock_data: Get stock data for a specified time frame for trend analysis. The time frame options are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max. 
            """,
        ),
        model = ModelFactory.create(
            model_platform="openai",  
            model_type="gpt-4o",   
        ),
        tools=stock_tools,
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
            model_type="gpt-4o",
        ),
    )
    planner = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Task Planner",
            content="Your job is to decompose tasks and compose results."
        ),
        model = ModelFactory.create(
            model_platform="openai",
            model_type="gpt-4o",
        ),
    )
        
    workforce = Workforce(
        "Financial Analyst",
        coordinator_agent = coordinator,
        task_agent = planner,
    )

    workforce.add_single_agent_worker(
        description = "worker agent for web search ",
        worker = search_agent,
    ).add_single_agent_worker(
        description = "worker agent for stock information",
        worker = stock_agent,
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

def app(input_question: str) -> dict:
    return multi_agent_with_verifier(input_question)


