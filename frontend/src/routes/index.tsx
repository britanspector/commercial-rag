import { createBrowserRouter } from 'react-router-dom'

import { AppLayout } from '../layouts/AppLayout'
import { ChatPage } from '../pages/chat/ChatPage'
import { CachePage } from '../pages/cache/CachePage'
import { EvalPage } from '../pages/eval/EvalPage'
import { HomePage } from '../pages/HomePage'
import { SearchPage } from '../pages/search/SearchPage'
import { TracePage } from '../pages/trace/TracePage'
import { UploadPage } from '../pages/upload/UploadPage'
import { WorkflowPage } from '../pages/workflow/WorkflowPage'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      { index: true, element: <HomePage /> },
      { path: 'upload', element: <UploadPage /> },
      { path: 'chat', element: <ChatPage /> },
      { path: 'search', element: <SearchPage /> },
      { path: 'eval', element: <EvalPage /> },
      { path: 'cache', element: <CachePage /> },
      { path: 'trace', element: <TracePage /> },
      { path: 'workflow', element: <WorkflowPage /> },
    ],
  },
])
