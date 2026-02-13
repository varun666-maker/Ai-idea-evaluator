from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from google import genai
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types
import asyncio
import json
import re

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

os.environ["GEMINI_API_KEY"] = API_KEY

# Initialize FastAPI app
app = FastAPI(title="AI Idea Evaluator")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini client
client = genai.Client()

# Configure retry options for agent
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

# Initialize competitor finder agent
competitor_agent = Agent(
    name="competitor_finder",
    model=Gemini(
        model="gemini-2.0-flash",
        retry_options=retry_config
    ),
    description="An agent that finds competitors for a given business idea.",
    instruction="You are an expert business analyst. Find the top 5 competitors for the given idea. Use Google Search to find real companies. Return as a JSON array with objects containing 'name' and 'description' fields.",
    tools=[google_search]
)

competitor_runner = InMemoryRunner(agent=competitor_agent)

# ==================== REQUEST MODELS ====================
class IdeaInput(BaseModel):
    idea: str
    location: str = None

class EvaluationRequest(BaseModel):
    idea: str
    location: str = None

# ==================== RESPONSE MODELS ====================

# 1. SWOT Analysis
class SWOTAnalysis(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    opportunities: list[str]
    threats: list[str]

# 2. Competitor Analysis
class Competitor(BaseModel):
    name: str
    description: str

class CompetitorAnalysis(BaseModel):
    competitors: list[Competitor]

# 3. Market Analysis (TAM, SAM, SOM) - NUMBERS ONLY
class MarketAnalysis(BaseModel):
    tam: str
    sam: str
    som: str

# 4. Market Size & Feasibility
class MarketSizeAnalysis(BaseModel):
    market_size_score: float
    market_size_description: str
    potential_score: float
    potential_description: str
    feasibility_score: float
    feasibility_description: str

# 5. Overall Evaluation with Brutal Honesty
class Risk(BaseModel):
    risk_name: str
    risk_description: str

class OverallEvaluationScore(BaseModel):
    overall_viability_score: float
    overall_risk_score: float  # 0-100, single overall risk score
    verdict: str
    verdict_description: str
    key_risks: list[Risk]

# Complete Response
class OverallEvaluation(BaseModel):
    swot: SWOTAnalysis
    competitors: CompetitorAnalysis
    market_analysis: MarketAnalysis
    market_size: MarketSizeAnalysis
    overall_evaluation: OverallEvaluationScore

# ==================== HELPER FUNCTIONS ====================

def extract_agent_text(response) -> str:
    """Extract text from agent response"""
    response_str = str(response)
    
    # Try to extract text between triple quotes
    pattern = r'text="""(.*?)"""'
    matches = re.findall(pattern, response_str, re.DOTALL)
    
    if matches:
        return matches[0]
    
    # Try alternative pattern
    pattern2 = r'text="(.*?)(?:\n|")'
    matches2 = re.findall(pattern2, response_str, re.DOTALL)
    if matches2:
        return matches2[0]
    
    return ""

def parse_list_response(text: str, count: int = 3) -> list[str]:
    """Parse numbered list response"""
    items = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Remove numbering
        match = re.match(r'^\d+\.\s+(.+)$', line)
        if match:
            cleaned = match.group(1).strip()
            if cleaned and len(cleaned) > 1:
                items.append(cleaned)
    
    return items[:count]

# ==================== ENDPOINTS ====================

@app.post("/evaluate-idea", response_model=OverallEvaluation)
async def evaluate_idea(request: EvaluationRequest):
    """
    Comprehensive AI Idea Evaluator
    Performs SWOT, Competitor, Market, and Feasibility Analysis
    """
    try:
        idea = request.idea
        location = request.location or "global"
        
        print(f"\n{'='*60}")
        print(f"EVALUATING IDEA: {idea}")
        print(f"LOCATION: {location}")
        print(f"{'='*60}\n")
        
        # ===== 1. SWOT ANALYSIS =====
        print("1. Performing SWOT Analysis...")
        swot = await _perform_swot_analysis(idea, location)
        print("✓ SWOT Analysis Complete\n")
        
        # ===== 2. COMPETITOR ANALYSIS =====
        print("2. Performing Competitor Analysis...")
        competitors = await _perform_competitor_analysis(idea, location)
        print("✓ Competitor Analysis Complete\n")
        
        # ===== Get target audience for market analysis =====
        target_audience = await _identify_target_audience(idea, location)
        
        # ===== 3. MARKET ANALYSIS (TAM, SAM, SOM) =====
        print("3. Performing Market Analysis...")
        market = await _perform_market_analysis(idea, location, len(competitors.competitors))
        print("✓ Market Analysis Complete\n")
        
        # ===== 4. MARKET SIZE & FEASIBILITY =====
        print("4. Analyzing Market Size & Feasibility...")
        market_size = await _analyze_market_size(idea, location)
        print("✓ Market Size Analysis Complete\n")
        
        # ===== OVERALL EVALUATION =====
        print("5. Calculating Overall Evaluation...")
        overall_evaluation = calculate_overall_evaluation(swot, competitors, market, market_size)
        print("✓ Overall Evaluation Complete\n")
        
        print(f"\n{'='*60}")
        print(f"OVERALL VIABILITY SCORE: {overall_evaluation.overall_viability_score:.2f}/100")
        print(f"VERDICT: {overall_evaluation.verdict}")
        print(f"{'='*60}\n")
        
        return OverallEvaluation(
            swot=swot,
            competitors=competitors,
            market_analysis=market,
            market_size=market_size,
            overall_evaluation=overall_evaluation
        )
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error evaluating idea: {str(e)}")


async def _perform_swot_analysis(idea: str, location: str) -> SWOTAnalysis:
    """Perform SWOT Analysis"""
    try:
        # Strengths
        prompt = f"For this business idea: '{idea}' in {location}, list exactly 3 key STRENGTHS (competitive advantages). Keep each point under 35 words. Return only the 3 strengths as a numbered list (1. 2. 3.), nothing else."
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        strengths = parse_list_response(response.text, 3)
        
        # Weaknesses
        prompt = f"For this business idea: '{idea}' in {location}, list exactly 3 key WEAKNESSES (internal challenges). Keep each point under 35 words. Return only the 3 weaknesses as a numbered list (1. 2. 3.), nothing else."
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        weaknesses = parse_list_response(response.text, 3)
        
        # Opportunities
        prompt = f"For this business idea: '{idea}' in {location}, list exactly 3 OPPORTUNITIES (market growth potential). Keep each point under 35 words. Return only the 3 opportunities as a numbered list (1. 2. 3.), nothing else."
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        opportunities = parse_list_response(response.text, 3)
        
        # Threats
        prompt = f"For this business idea: '{idea}' in {location}, list exactly 3 THREATS (external risks). Keep each point under 35 words. Return only the 3 threats as a numbered list (1. 2. 3.), nothing else."
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        threats = parse_list_response(response.text, 3)
        
        return SWOTAnalysis(
            strengths=strengths,
            weaknesses=weaknesses,
            opportunities=opportunities,
            threats=threats
        )
    except Exception as e:
        print(f"Error in SWOT: {e}")
        raise

async def _perform_competitor_analysis(idea: str, location: str) -> CompetitorAnalysis:
    """Perform Competitor Analysis using Agent"""
    try:
        # Step 1: Get competitor names
        search_prompt = f"List the top 5 direct competitors/companies for '{idea}' . Use Google Search. Return ONLY the company names as a numbered list (1. 2. 3. 4. 5.), nothing else. Just names, no explanations."
        
        response = await competitor_runner.run_debug(search_prompt)
        competitor_text = extract_agent_text(response)
        
        print(f"Raw competitor text: {competitor_text}")
        
        # Parse competitor names - only get clean names
        competitors_list = []
        lines = parse_list_response(competitor_text, 5)
        
        print(f"Parsed competitor names: {lines}")
        
        # Step 2: Get descriptions for each competitor (max 70 words)
        for name in lines:
            # Clean up the name
            clean_name = name.strip()
            if not clean_name:
                continue
            
            # Get specific description for this competitor
            desc_prompt = f"In 1 sentence (max 35 words), describe what {clean_name} does in the {idea} space. Be concise and factual."
            desc_response = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=desc_prompt
            )
            description = desc_response.text.strip()
            # Truncate to 35 words if needed
            words = description.split()
            if len(words) > 35:
                description = ' '.join(words[:35])
            
            competitors_list.append(Competitor(
                name=clean_name,
                description=description
            ))
        
        # Determine competition level based on count
        competition_count = len(competitors_list)
        
        print(f"Found {competition_count} competitors")
        
        return CompetitorAnalysis(
            competitors=competitors_list
        )
    except Exception as e:
        print(f"Error in Competitor Analysis: {e}")
        import traceback
        traceback.print_exc()
        return CompetitorAnalysis(
            competitors=[],
            competition_level="Unknown",
            competition_score=50.0
        )

async def _perform_market_analysis(idea: str, location: str, competitors_count: int = 5) -> MarketAnalysis:
    """Perform Market Analysis (TAM, SAM, SOM) with data-driven calculations"""
    try:
        # Step 1: Get target audience size
        prompt = f"For '{idea}' in {location}, estimate the total number of potential customers in the target audience. Provide just a number or range (e.g., '5 million', '100k-500k'). Consider the entire market in {location}."
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        audience_size_text = response.text.strip()
        
        # Extract number from response
        audience_size = extract_number_from_text(audience_size_text)
        print(f"Target audience size: {audience_size_text} (~{audience_size:,.0f} people)")
        
        # Step 2: Get average monthly pricing
        prompt = f"For '{idea}', what would be a reasonable monthly subscription price or average revenue per user (ARPU)? Give a single price in USD (e.g., '$29/month' or '$49/month'). Consider market positioning."
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        pricing_text = response.text.strip()
        price_per_month = extract_price_from_text(pricing_text)
        print(f"Average monthly price: {pricing_text} (~${price_per_month}/month)")
        
        # Step 3: Calculate TAM (Total Addressable Market)
        annual_price = price_per_month * 12
        tam_value = audience_size * annual_price
        tam_display = format_currency(tam_value)
        print(f"TAM: {tam_display}")
        
        # Step 4: Get SAM based on competition
        prompt = f"For '{idea}', what percentage of the total market could realistically be captured considering it's a new entrant? (Give a percentage like '10%', '15%', etc.)"
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        sam_percentage_text = response.text.strip()
        sam_percentage = extract_percentage_from_text(sam_percentage_text)
        
        # Adjust SAM based on competition level
        if competitors_count >= 5:  # High competition
            competition_adjustment = 0.6
            sam_percentage *= competition_adjustment
        elif competitors_count >= 3:  # Medium competition
            competition_adjustment = 0.75
            sam_percentage *= competition_adjustment
        else:  # Low competition
            competition_adjustment = 0.9
            sam_percentage *= competition_adjustment
        
        sam_value = tam_value * (sam_percentage / 100)
        sam_display = format_currency(sam_value)
        print(f"SAM: {sam_display}")
        
        # Step 5: Get SOM (realistic 5-year market capture)
        prompt = f"For a new '{idea}' startup, what's a realistic market share in 5 years? (Give a percentage like '2%', '5%', '1%'. Be realistic for a new entrant.)"
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        som_percentage_text = response.text.strip()
        som_percentage = extract_percentage_from_text(som_percentage_text)
        
        som_value = tam_value * (som_percentage / 100)
        som_display = format_currency(som_value)
        print(f"SOM: {som_display}")
        
        return MarketAnalysis(
            tam=tam_display,
            sam=sam_display,
            som=som_display
        )
    except Exception as e:
        print(f"Error in Market Analysis: {e}")
        import traceback
        traceback.print_exc()
        raise

def extract_number_from_text(text: str) -> float:
    """Extract number from text like '5 million', '100k', etc."""
    text = text.lower().strip()
    
    # Handle millions
    if 'million' in text or 'm' in text:
        numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
        if numbers:
            return float(numbers[0]) * 1_000_000
    
    # Handle thousands
    if 'thousand' in text or 'k' in text:
        numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
        if numbers:
            return float(numbers[0]) * 1_000
    
    # Handle plain numbers
    numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
    if numbers:
        return float(numbers[0])
    
    # Default estimate
    return 1_000_000

def extract_price_from_text(text: str) -> float:
    """Extract price from text like '$29/month' or '49'"""
    numbers = re.findall(r'\$?(\d+(?:\.\d+)?)', text)
    if numbers:
        return float(numbers[0])
    return 49.0  # Default

def extract_percentage_from_text(text: str) -> float:
    """Extract percentage from text like '10%' or 'ten percent'"""
    # Look for explicit percentage
    percentages = re.findall(r'(\d+(?:\.\d+)?)%', text)
    if percentages:
        return float(percentages[0])
    
    # Default
    return 10.0

def format_currency(value: float) -> str:
    """Format large numbers as currency"""
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.2f}K"
    else:
        return f"${value:.2f}"

async def _analyze_market_size(idea: str, location: str) -> MarketSizeAnalysis:
    """Analyze Market Size, Potential, and Feasibility with scores and descriptions"""
    try:
        # MARKET SIZE
        prompt = f"Is the market size for '{idea}' in {location} Small, Medium, or Large? Explain briefly (max 35 words)."
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        size_response = response.text.strip()
        size_desc = truncate_to_words(size_response, 35)
        size_score = 75.0 if "large" in size_response.lower() else (50.0 if "medium" in size_response.lower() else 25.0)
        
        # POTENTIAL RATING
        prompt = f"What is the revenue/growth potential for '{idea}' in {location}? (High, Medium, or Low). Explain briefly (max 35 words)."
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        potential_response = response.text.strip()
        potential_desc = truncate_to_words(potential_response, 35)
        potential_score = 75.0 if "high" in potential_response.lower() else (50.0 if "medium" in potential_response.lower() else 25.0)
        
        # FEASIBILITY
        prompt = f"What is the feasibility (technical, financial, market) for implementing '{idea}' in {location}? (High, Medium, or Low). Explain briefly (max 35 words)."
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        feasibility_response = response.text.strip()
        feasibility_desc = truncate_to_words(feasibility_response, 35)
        feasibility_score = 75.0 if "high" in feasibility_response.lower() else (50.0 if "medium" in feasibility_response.lower() else 25.0)
        
        return MarketSizeAnalysis(
            market_size_score=size_score,
            market_size_description=size_desc,
            potential_score=potential_score,
            potential_description=potential_desc,
            feasibility_score=feasibility_score,
            feasibility_description=feasibility_desc
        )
    except Exception as e:
        print(f"Error in Market Size Analysis: {e}")
        raise

async def _identify_target_audience(idea: str, location: str) -> tuple:
    """Identify Target Audience - returns tuple to avoid including in output"""
    try:
        prompt = f"For '{idea}' in {location}, who is the PRIMARY target audience? (1 sentence)"
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        primary = response.text.strip()
        
        return (primary,)  # Return as tuple so it's not part of output
    except Exception as e:
        print(f"Error in Target Audience: {e}")
        return ("Unknown",)

def truncate_to_words(text: str, max_words: int) -> str:
    """Truncate text to max words"""
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return text

def calculate_overall_evaluation(swot: SWOTAnalysis, competitors: CompetitorAnalysis, 
                                 market: MarketAnalysis, market_size: MarketSizeAnalysis) -> OverallEvaluationScore:
    """Calculate overall viability score with BRUTAL HONEST verdict based on all parameters"""
    
    # Determine competition score based on number of competitors
    competition_count = len(competitors.competitors)
    if competition_count <= 2:
        competition_score = 25.0
    elif competition_count <= 4:
        competition_score = 50.0
    else:
        competition_score = 75.0
    
    # Calculate individual scores
    swot_score = 70.0  # Base SWOT score
    competitor_weighted = (100 - competition_score) * 0.20
    market_weighted = 75.0 * 0.25  # Using default market score
    market_size_weighted = (market_size.market_size_score + market_size.potential_score + market_size.feasibility_score) / 3 * 0.30
    
    overall_score = (swot_score * 0.25) + competitor_weighted + market_weighted + market_size_weighted
    overall_score = min(100, max(0, overall_score))
    
    print(f"\nDEBUG - Viability Score: {overall_score:.2f}")
    print(f"DEBUG - Competition Score: {competition_score}")
    print(f"DEBUG - Market Size Score: {market_size.market_size_score}, Potential: {market_size.potential_score}, Feasibility: {market_size.feasibility_score}")
    
    # ===== IDENTIFY KEY BUSINESS RISKS =====
    high_competition = competition_score >= 70
    low_potential = market_size.potential_score <= 40
    poor_feasibility = market_size.feasibility_score <= 40
    small_market = market_size.market_size_score <= 40
    
    print(f"DEBUG - High Competition: {high_competition}, Low Potential: {low_potential}, Poor Feasibility: {poor_feasibility}, Small Market: {small_market}")
    
    # Calculate overall risk score (0-100)
    risk_factors = []
    
    # Risk 1: Market Competition
    if high_competition:
        risk_factors.append(Risk(
            risk_name="High Market Competition",
            risk_description="Saturated market with established competitors offering similar solutions."
        ))
    
    # Risk 2: SWOT Threats
    if swot.threats and len(swot.threats) > 0:
        risk_factors.append(Risk(
            risk_name="Market Threats",
            risk_description=swot.threats[0][:50]
        ))
    
    # Risk 3: Limited Market Demand
    if low_potential:
        risk_factors.append(Risk(
            risk_name="Low Market Demand",
            risk_description="Limited awareness and adoption potential in target geography."
        ))
    
    # Risk 4: Feasibility Challenges
    if poor_feasibility:
        risk_factors.append(Risk(
            risk_name="Execution Feasibility",
            risk_description=swot.weaknesses[0][:50] if swot.weaknesses else "Technical challenges."
        ))
    
    # Risk 5: Small Market Size
    if small_market:
        risk_factors.append(Risk(
            risk_name="Limited Market Size",
            risk_description="Market is geographically limited, constraining growth potential."
        ))
    
    # Calculate overall risk score based on identified risks
    overall_risk_score = 0.0
    if risk_factors:
        # Each identified risk contributes to overall risk
        risk_contribution = 20.0  # Each risk adds 20 points
        overall_risk_score = min(100.0, len(risk_factors) * risk_contribution)
    
    # Ensure we always have at least one risk
    if not risk_factors:
        risk_factors.append(Risk(
            risk_name="General Market Risk",
            risk_description="General execution and market validation risks present."
        ))
        overall_risk_score = 30.0
    
    print(f"DEBUG - Risks found: {len(risk_factors)}, Overall Risk Score: {overall_risk_score:.1f}")
    
    # ===== DETERMINE VERDICT =====
    verdict = ""
    verdict_desc = ""
    
    if overall_score >= 85 and high_competition == False:
        verdict = "EXCELLENT IDEA"
        verdict_desc = f"Strong market fundamentals with viability score {overall_score:.1f}/100. Solid competitive advantages, manageable risks. High growth potential with viable execution path. Proceed with confidence."
    
    elif overall_score >= 70 and competition_score <= 65:
        verdict = "GOOD IDEA"
        verdict_desc = f"Solid opportunity with viability score {overall_score:.1f}/100. Market demand exists, differentiation possible. Competition manageable with proper execution. Worth pursuing with strong planning."
    
    elif overall_score >= 55 and not poor_feasibility:
        verdict = "MODERATE IDEA"
        verdict_desc = f"Mixed signals with viability score {overall_score:.1f}/100. Potential exists but significant challenges need resolution. Requires thorough market validation and strong differentiation strategy before heavy investment."
    
    elif overall_score >= 40:
        verdict = "WEAK IDEA"
        verdict_desc = f"Significant concerns with viability score {overall_score:.1f}/100. Market is saturated or not validated. Major weaknesses and high threats. Deep pivoting or major refinements needed before proceeding."
    
    else:
        verdict = "BAD IDEA"
        verdict_desc = f"Do not pursue - viability score is only {overall_score:.1f}/100. Multiple fundamental issues: saturated market, weak differentiation, low demand, poor feasibility. Resources better spent on other opportunities."
    
    print(f"DEBUG - Verdict: {verdict}, Overall Risk Score: {overall_risk_score:.1f}")
    
    return OverallEvaluationScore(
        overall_viability_score=overall_score,
        overall_risk_score=overall_risk_score,
        verdict=verdict,
        verdict_description=verdict_desc,
        key_risks=risk_factors
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Idea Evaluator"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)