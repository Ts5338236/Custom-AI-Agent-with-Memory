from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.orm import Session
from app.models.chat import UserNode, UserEdge
from app.core.config import settings
import json

class KnowledgeGraphService:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    async def extract_and_update(self, db: Session, user_id: int, text: str):
        """
        Extracts entities and relationships from text and updates the graph.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract entities and relationships from the text. "
                       "Return a JSON list of objects: {'source': str, 'target': str, 'relationship': str}. "
                       "Example: John works at Acme -> {'source': 'John', 'target': 'Acme', 'relationship': 'WORKS_AT'}"),
            ("human", "{text}")
        ])
        
        try:
            response = await (prompt | self.llm).ainvoke({"text": text})
            # Naive parse (in production use structured output)
            triples = json.loads(response.content.replace("```json", "").replace("```", "").strip())
            
            for triple in triples:
                # 1. Ensure nodes exist
                source = self._get_or_create_node(db, user_id, triple['source'])
                target = self._get_or_create_node(db, user_id, triple['target'])
                
                # 2. Create edge
                edge = UserEdge(
                    user_id=user_id,
                    source_id=source.id,
                    target_id=target.id,
                    relationship=triple['relationship']
                )
                db.add(edge)
            db.commit()
        except Exception as e:
            print(f"Graph Extraction Error: {e}")

    def _get_or_create_node(self, db: Session, user_id: int, name: str) -> UserNode:
        node = db.query(UserNode).filter(UserNode.user_id == user_id, UserNode.name == name).first()
        if not node:
            node = UserNode(user_id=user_id, name=name, label="Entity")
            db.add(node)
            db.flush()
        return node

    def get_related_context(self, db: Session, user_id: int, entities: list[str]) -> str:
        """
        Traverses the graph to find connected information for the mentioned entities.
        """
        # Find edges where any of these entities are source or target
        nodes = db.query(UserNode).filter(UserNode.user_id == user_id, UserNode.name.in_(entities)).all()
        node_ids = [n.id for n in nodes]
        
        edges = db.query(UserEdge).filter(
            UserEdge.user_id == user_id,
            (UserEdge.source_id.in_(node_ids)) | (UserEdge.target_id.in_(node_ids))
        ).all()
        
        context = []
        for edge in edges:
            source = db.query(UserNode).get(edge.source_id).name
            target = db.query(UserNode).get(edge.target_id).name
            context.append(f"- {source} {edge.relationship} {target}")
            
        return "\n".join(context)

knowledge_graph_service = KnowledgeGraphService()
