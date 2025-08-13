from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from euriai.langchain import EuriaiChatModel
from euriai import EuriaiClient
import os
from typing import List, Dict

class FPLKnowledgeBase:
    def __init__(self, euri_api_key: str, model: str = "gemini-2.5-pro"):
        self.euri_api_key = os.getenv("EURI_API_KEY")
        self.model = model
        
        # Initialize Euri AI client for direct API calls
        self.client = EuriaiClient(
            api_key=euri_api_key,
            model=model
        )
        
        # Initialize LangChain integration
        try:
            self.chat_model = EuriaiChatModel(
                api_key=euri_api_key,
                model=model,
                temperature=0.1  # Lower temperature for more consistent strategic advice
            )
        except ImportError as e:
            print(f"LangChain integration not available: {e}")
            self.chat_model = None
        
        self.vectorstore = None
        self.qa_chain = None
        
    def create_knowledge_base(self, documents: List[str]):
        """Create vector database from FPL knowledge documents"""
        if not self.chat_model:
            print("Warning: LangChain integration not available. RAG system will use direct API calls.")
            return
            
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separator="\n\n"
        )
        texts = text_splitter.create_documents(documents)
        
        # Use Chroma with default embeddings (or implement Euri AI embeddings if available)
        try:
            self.vectorstore = Chroma.from_documents(
                texts,
                persist_directory="./fpl_knowledge_db"
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.chat_model,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
            )
            print("RAG knowledge base initialized successfully")
        except Exception as e:
            print(f"Error initializing RAG system: {e}")
    
    def get_strategic_advice(self, query: str) -> str:
        """Get strategic advice using RAG or direct API call"""
        if self.qa_chain:
            try:
                return self.qa_chain.run(query)
            except Exception as e:
                print(f"RAG query failed, falling back to direct API: {e}")
        
        # Fallback to direct API call with context
        context = self._get_relevant_context(query)
        enhanced_prompt = f"""
        Based on Fantasy Premier League strategy and the following context:
        
        {context}
        
        User Question: {query}
        
        Please provide strategic advice for Fantasy Premier League management.
        Consider factors like:
        - Fixture difficulty and scheduling
        - Player form and injury status
        - Budget management
        - Chip usage timing
        - Risk vs reward balance
        
        Response:"""
        
        try:
            response = self.client.generate_completion(
                prompt=enhanced_prompt,
                temperature=0.3,
                max_tokens=500
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error getting strategic advice: {str(e)}"
    
    def _get_relevant_context(self, query: str) -> str:
        """Get relevant context for queries when RAG is not available"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['wildcard', 'chip', 'free hit', 'bench boost', 'triple captain']):
            return FPL_CHIP_STRATEGY_CONTEXT
        elif any(word in query_lower for word in ['captain', 'captaincy']):
            return FPL_CAPTAIN_STRATEGY_CONTEXT
        elif any(word in query_lower for word in ['transfer', 'budget', 'price']):
            return FPL_TRANSFER_STRATEGY_CONTEXT
        else:
            return FPL_GENERAL_STRATEGY_CONTEXT
    
    def analyze_chip_strategy(self, gameweek: int, fixtures: List[Dict], available_chips: List[str]) -> ChipRecommendation:
        """Analyze when to use chips based on fixtures and historical data"""
        
        # Prepare context about upcoming fixtures
        fixture_context = self._analyze_fixtures(fixtures, gameweek)
        
        query = f"""
        Current gameweek: {gameweek}
        Available chips: {', '.join(available_chips)}
        
        Fixture analysis: {fixture_context}
        
        Based on the upcoming fixtures and current gameweek, which chip should I prioritize using and when?
        Consider:
        1. Double gameweeks and blank gameweeks
        2. Fixture difficulty for top teams
        3. International breaks and cup schedules
        4. Optimal timing for maximum points
        
        Provide a specific recommendation with reasoning.
        """
        
        advice = self.get_strategic_advice(query)
        
        # Parse the advice to create a structured recommendation
        # This is a simplified parsing - in production, you might use more sophisticated NLP
        recommended_chip = self._extract_chip_from_advice(advice, available_chips)
        confidence = self._estimate_confidence(advice)
        expected_benefit = self._estimate_benefit(recommended_chip, gameweek)
        
        return ChipRecommendation(
            chip=ChipType(recommended_chip),
            recommended_gameweek=gameweek + self._get_gameweek_offset(recommended_chip),
            confidence=confidence,
            reasoning=advice,
            expected_benefit=expected_benefit
        )
    
    def _analyze_fixtures(self, fixtures: List[Dict], current_gameweek: int) -> str:
        """Analyze upcoming fixtures for strategic insights"""
        # This would analyze fixture difficulty, double gameweeks, etc.
        # Simplified version for demo
        upcoming_fixtures = [f for f in fixtures if f.get('event', 0) >= current_gameweek][:20]
        
        analysis = f"Next 5 gameweeks analysis:\n"
        
        # Group fixtures by gameweek
        gw_fixtures = {}
        for fixture in upcoming_fixtures:
            gw = fixture.get('event', current_gameweek)
            if gw not in gw_fixtures:
                gw_fixtures[gw] = []
            gw_fixtures[gw].append(fixture)
        
        for gw in sorted(gw_fixtures.keys())[:5]:
            fixture_count = len(gw_fixtures[gw])
            analysis += f"GW{gw}: {fixture_count} fixtures\n"
        
        return analysis
    
    def _extract_chip_from_advice(self, advice: str, available_chips: List[str]) -> str:
        """Extract recommended chip from AI advice"""
        advice_lower = advice.lower()
        
        chip_priorities = {
            'triple_captain': ['triple captain', 'tc', 'captain'],
            'bench_boost': ['bench boost', 'bb', 'bench'],
            'free_hit': ['free hit', 'fh'],
            'wildcard': ['wildcard', 'wc']
        }
        
        for chip, keywords in chip_priorities.items():
            if chip in available_chips and any(keyword in advice_lower for keyword in keywords):
                return chip
        
        # Default to first available chip if none specifically mentioned
        return available_chips[0] if available_chips else 'wildcard'
    
    def _estimate_confidence(self, advice: str) -> float:
        """Estimate confidence level from advice text"""
        confidence_indicators = {
            'definitely': 0.9,
            'strongly recommend': 0.85,
            'should': 0.8,
            'recommend': 0.75,
            'consider': 0.6,
            'might': 0.5,
            'could': 0.45
        }
        
        advice_lower = advice.lower()
        max_confidence = 0.5  # Default confidence
        
        for indicator, confidence in confidence_indicators.items():
            if indicator in advice_lower:
                max_confidence = max(max_confidence, confidence)
        
        return max_confidence
    
    def _estimate_benefit(self, chip: str, gameweek: int) -> float:
        """Estimate expected benefit points for using a chip"""
        chip_benefits = {
            'triple_captain': 15.0,
            'bench_boost': 12.0,
            'free_hit': 20.0,
            'wildcard': 18.0
        }
        
        base_benefit = chip_benefits.get(chip, 10.0)
        
        # Adjust based on timing (early season vs late season)
        if gameweek <= 10:
            multiplier = 1.2  # Higher uncertainty early
        elif gameweek >= 30:
            multiplier = 0.9   # Less impact late season
        else:
            multiplier = 1.0
        
        return base_benefit * multiplier
    
    def _get_gameweek_offset(self, chip: str) -> int:
        """Get recommended gameweek offset for chip usage"""
        chip_timing = {
            'wildcard': 1,      # Use soon for team restructuring
            'free_hit': 3,      # Wait for blank gameweek
            'bench_boost': 2,   # Wait for double gameweek
            'triple_captain': 2 # Wait for favorable fixtures
        }
        return chip_timing.get(chip, 1)

# Enhanced FPL knowledge base with more detailed strategic content
FPL_GENERAL_STRATEGY_CONTEXT = """
Fantasy Premier League General Strategy:
- Budget management: Keep 0.5-1.0m in bank for price rises
- Team structure: Invest in premium players who consistently deliver
- Fixture planning: Plan transfers 3-4 gameweeks ahead
- Risk management: Balance template picks with differentials
"""

FPL_CHIP_STRATEGY_CONTEXT = """
FPL Chip Strategy:
- Wildcard: Use during international breaks or major team overhaul needed
- Free Hit: Best for blank gameweeks when many players don't play
- Bench Boost: Use during double gameweeks when bench players also have fixtures
- Triple Captain: Save for double gameweeks of premium players with good fixtures
"""

FPL_CAPTAIN_STRATEGY_CONTEXT = """
FPL Captaincy Strategy:
- Always captain players with highest expected points
- Consider fixture difficulty rating (FDR)
- Home fixtures generally better for attacking returns
- Premium players vs weaker defenses are ideal
- Monitor team news for injuries and rotation
"""

FPL_TRANSFER_STRATEGY_CONTEXT = """
FPL Transfer Strategy:
- Don't take hits unless expecting 4+ point gain
- Prioritize injured/suspended players for transfers
- Consider price changes and ownership trends
- Plan for fixture swings and double gameweeks
- Use free transfers efficiently each week
"""

# Initialize FPL knowledge base with strategic content
FPL_KNOWLEDGE = [
    """
    Fantasy Premier League Strategy Guide:

    Always use the latest FPL data from the latest 2025-2026 English Premier League.
    
    Captain Selection: Always captain players with the highest expected points.
    Consider fixture difficulty, form, and historical performance.
    
    Chip Strategy:
    - Wildcard: Use during international breaks or when you need major team changes
    - Free Hit: Best used in double gameweeks when many of your players blank
    - Bench Boost: Use during double gameweeks when your bench players also have fixtures
    - Triple Captain: Save for double gameweeks of premium players
    
    Budget Management:
    - Keep 0.5-1.0m in the bank for price rises
    - Invest heavily in premium players who consistently deliver
    - Find budget gems in defense and midfield
    """,
    
    """
    Position-Specific Strategies:
    
    Goalkeepers:
    - Rotate between two 4.5m goalkeepers
    - Focus on clean sheet potential and save points
    
    Defenders:
    - Premium defenders (6m+) from top 6 teams for attacking returns
    - Budget defenders from teams with good defensive records
    - Target players who take set pieces
    
    Midfielders:
    - Most important position for points
    - Target players classified as midfielders who play as forwards
    - Consider penalty takers and set piece specialists
    
    Forwards:
    - Premium forwards are essential for captaincy
    - Budget forwards should be enablers for premium players elsewhere
    """,
    
    """
    Fixture Analysis:
    
    Double Gameweeks:
    - Players get two fixtures in one gameweek
    - Perfect time to use Triple Captain and Bench Boost
    - Target players from teams with favorable double fixtures
    
    Blank Gameweeks:
    - Some teams don't play due to cup competitions
    - Use Free Hit chip to field a full team
    - Plan transfers to avoid blank players
    
    Fixture Difficulty Rating (FDR):
    - 1-2: Very favorable fixtures
    - 3: Average fixtures  
    - 4-5: Difficult fixtures
    """
]