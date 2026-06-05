import { NavPage } from '../../components/NavPage'
import { getNavItemByPath } from '../../navigation'

export function ChatPage() {
  const item = getNavItemByPath('/chat')!
  return <NavPage item={item} />
}
