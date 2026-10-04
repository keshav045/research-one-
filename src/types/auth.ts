/** A signed-in user stored in localStorage. */
export interface User {
  id: string;
  name: string;
  email: string;
  /** Initials derived from name, used as avatar fallback. */
  initials: string;
  /** Hex color for the avatar background, deterministic from the user id. */
  avatarColor: string;
  createdAt: string;
}

export interface AuthState {
  user: User | null;
  isLoading: boolean;
}
