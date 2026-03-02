export interface BookInfo {
  title: string | null;
  author: string | null;
  isbn: string | null;
  publisher: string | null;
  publication_year: number | null;
  genre: string | null;
  tags: string[];
  synopsis: string | null;
  price: number | null;
  rating: number | null;
  confidence_score: number;
}

export interface DetectionResponse {
  success: boolean;
  book: BookInfo;
  extraction_notes: string | null;
}
